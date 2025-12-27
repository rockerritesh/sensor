"""
Flask backend API for real-time hand detection
Serves predictions using the trained ML model with temporal smoothing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from collections import deque
from scipy.stats import mode
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for HTML frontend

# Load models
print("Loading models...")
with open('hand_classifier_model.pkl', 'rb') as f:
    ml_model = pickle.load(f)
with open('hand_classifier_scaler.pkl', 'rb') as f:
    ml_scaler = pickle.load(f)
with open('pca_model.pkl', 'rb') as f:
    pca_model = pickle.load(f)
with open('scaler_pca.pkl', 'rb') as f:
    scaler_pca = pickle.load(f)

# Try to load windowed model if available
windowed_model = None
windowed_scaler = None
window_params = None

if os.path.exists('hand_classifier_windowed_model.pkl'):
    print("Loading windowed model...")
    with open('hand_classifier_windowed_model.pkl', 'rb') as f:
        windowed_model = pickle.load(f)
    with open('hand_classifier_windowed_scaler.pkl', 'rb') as f:
        windowed_scaler = pickle.load(f)
    with open('window_params.pkl', 'rb') as f:
        window_params = pickle.load(f)
    print("Windowed model loaded successfully!")
else:
    print("Windowed model not found. Using single-point predictions only.")

print("Models loaded successfully!")

# Global buffer for temporal smoothing
# Store separate buffers per session (using simple single buffer for now)
prediction_buffer = deque(maxlen=20)
WINDOW_SIZE = 20

def predict_hand_single(x, y, z):
    """
    Single-point prediction (no smoothing)
    Returns: hand, confidence, pc1, pc2
    """
    # Calculate magnitude
    magnitude = np.sqrt(x**2 + y**2 + z**2)

    # Create feature array
    features = np.array([[x, y, z, magnitude]])

    # Scale features
    features_scaled = ml_scaler.transform(features)

    # ML Prediction
    prediction_encoded = ml_model.predict(features_scaled)[0]
    prediction_proba = ml_model.predict_proba(features_scaled)[0]
    confidence = max(prediction_proba)

    # Map encoded labels to hand names
    # The model uses label encoding: 0 = left, 1 = right
    hand_label = 'left' if prediction_encoded == 0 else 'right'

    # PCA transformation
    features_pca_scaled = scaler_pca.transform(features)
    pca_coords = pca_model.transform(features_pca_scaled)
    pc1, pc2 = pca_coords[0]

    return {
        'hand': hand_label,
        'confidence': float(confidence),
        'pc1': float(pc1),
        'pc2': float(pc2),
        'prediction_encoded': int(prediction_encoded)
    }

def predict_hand_smoothed(x, y, z, mode='majority'):
    """
    Predict with temporal smoothing to prevent rapid switching

    Parameters:
    -----------
    x, y, z : float
        Accelerometer readings
    mode : str
        'majority' - majority vote over window
        'windowed' - use windowed feature model
        'confidence' - confidence-weighted voting

    Returns: hand, confidence, pc1, pc2, raw_prediction
    """
    # Add to buffer
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    prediction_buffer.append({
        'x': x, 'y': y, 'z': z, 'magnitude': magnitude
    })

    # Get single prediction for PCA (always needed for visualization)
    single_pred = predict_hand_single(x, y, z)

    # If buffer not full enough, return single prediction
    if len(prediction_buffer) < WINDOW_SIZE:
        return {
            **single_pred,
            'smoothed': False,
            'buffer_size': len(prediction_buffer)
        }

    # Apply smoothing based on mode
    if mode == 'windowed' and windowed_model is not None:
        smoothed_result = _predict_windowed()
    elif mode == 'confidence':
        smoothed_result = _predict_confidence_weighted()
    else:  # majority (default)
        smoothed_result = _predict_majority_vote()

    return {
        **single_pred,  # Keep PCA coords from single prediction
        'hand': smoothed_result['hand'],  # Override with smoothed prediction
        'confidence': smoothed_result['confidence'],
        'smoothed': True,
        'buffer_size': len(prediction_buffer),
        'raw_hand': single_pred['hand']  # Include raw prediction for comparison
    }

def _predict_majority_vote():
    """Majority vote over recent predictions"""
    predictions = []

    for reading in prediction_buffer:
        pred = predict_hand_single(reading['x'], reading['y'], reading['z'])
        predictions.append(pred['prediction_encoded'])

    # Get majority
    majority_pred = mode(predictions, keepdims=True)[0][0]
    confidence = sum(p == majority_pred for p in predictions) / len(predictions)

    hand = 'right' if majority_pred == 1 else 'left'
    return {'hand': hand, 'confidence': float(confidence)}

def _predict_confidence_weighted():
    """Confidence-weighted voting"""
    predictions = []
    confidences = []

    for reading in prediction_buffer:
        pred = predict_hand_single(reading['x'], reading['y'], reading['z'])
        predictions.append(pred['prediction_encoded'])
        confidences.append(pred['confidence'])

    # Weight votes by confidence
    right_weight = sum(c for p, c in zip(predictions, confidences) if p == 1)
    left_weight = sum(c for p, c in zip(predictions, confidences) if p == 0)
    total_weight = right_weight + left_weight

    if right_weight > left_weight:
        return {'hand': 'right', 'confidence': float(right_weight / total_weight)}
    else:
        return {'hand': 'left', 'confidence': float(left_weight / total_weight)}

def _predict_windowed():
    """Use windowed statistical features"""
    # Extract window features
    window_features = []
    buffer_list = list(prediction_buffer)

    for col in ['x', 'y', 'z', 'magnitude']:
        values = [reading[col] for reading in buffer_list]
        window_features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.median(values)
        ])

    # Add trends (delta)
    for col in ['x', 'y', 'z']:
        window_features.append(buffer_list[-1][col] - buffer_list[0][col])

    # Predict
    features = np.array([window_features])
    features_scaled = windowed_scaler.transform(features)

    prediction = windowed_model.predict(features_scaled)[0]
    proba = windowed_model.predict_proba(features_scaled)[0]

    hand = 'right' if prediction == 1 else 'left'
    confidence = float(proba[prediction])

    return {'hand': hand, 'confidence': confidence}

def predict_hand(x, y, z):
    """
    Main prediction function - uses smoothing by default
    """
    return predict_hand_smoothed(x, y, z, mode='majority')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for real-time predictions
    Expects JSON: {"x": float, "y": float, "z": float}
    Returns: {"hand": str, "confidence": float, "pc1": float, "pc2": float}
    """
    try:
        data = request.get_json()
        x = data['x']
        y = data['y']
        z = data['z']

        result = predict_hand(x, y, z)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    has_windowed = windowed_model is not None
    return jsonify({
        'status': 'ok',
        'message': 'Backend API is running',
        'features': {
            'windowed_model': has_windowed,
            'temporal_smoothing': True,
            'window_size': WINDOW_SIZE
        }
    })

@app.route('/clear_buffer', methods=['POST'])
def clear_buffer():
    """Clear the prediction buffer (useful when switching hands)"""
    prediction_buffer.clear()
    return jsonify({
        'status': 'ok',
        'message': 'Prediction buffer cleared',
        'buffer_size': len(prediction_buffer)
    })

@app.route('/predict_modes', methods=['GET'])
def get_predict_modes():
    """Get available prediction modes"""
    modes = ['majority', 'confidence']
    if windowed_model is not None:
        modes.append('windowed')

    return jsonify({
        'modes': modes,
        'default': 'majority',
        'descriptions': {
            'majority': 'Majority vote over last 20 predictions',
            'confidence': 'Confidence-weighted voting',
            'windowed': 'Statistical features over sliding window'
        }
    })

if __name__ == '__main__':
    PORT = 5001
    print("\n" + "="*50)
    print("🚀 Hand Detection Backend API with Temporal Smoothing")
    print("="*50)
    print(f"📍 Running on: http://localhost:{PORT}")
    print(f"📍 Prediction endpoint: http://localhost:{PORT}/predict")
    print(f"📍 Health check: http://localhost:{PORT}/health")
    print(f"📍 Clear buffer: http://localhost:{PORT}/clear_buffer")
    print(f"📍 Available modes: http://localhost:{PORT}/predict_modes")
    print("="*50)
    print(f"✨ Features:")
    print(f"   - Temporal smoothing: ENABLED")
    print(f"   - Window size: {WINDOW_SIZE} samples")
    print(f"   - Windowed model: {'LOADED' if windowed_model else 'Not available'}")
    print(f"   - Default mode: majority voting")
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=PORT, debug=True)
