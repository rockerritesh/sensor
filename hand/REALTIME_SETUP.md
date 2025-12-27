# Real-Time Hand Detection System

Complete setup for real-time hand detection using accelerometer data with ML predictions and PCA visualization.

## System Architecture

```
Sensor Device (Mobile) → WebSocket → HTML Frontend → HTTP API → Flask Backend (ML Model) → Predictions
                                          ↓
                                    PCA Visualization
```

## Files

- `backend_api.py` - Flask backend API serving ML predictions
- `realtime_hand_detector.html` - HTML/JavaScript frontend with real-time visualization
- `hand_classifier_model.pkl` - Trained Random Forest classifier
- `hand_classifier_scaler.pkl` - Feature scaler for ML model
- `pca_model.pkl` - PCA model for dimensionality reduction
- `scaler_pca.pkl` - Scaler for PCA features

## Setup Instructions

### 1. Install Dependencies

```bash
# Install Flask and Flask-CORS
pip install flask flask-cors

# Or if using uv
uv pip install flask flask-cors
```

### 2. Generate PCA Models (First Time Only)

If you don't have `pca_model.pkl` and `scaler_pca.pkl`:

```bash
# Using uv
uv run save_pca_models.py

# Or with python
python save_pca_models.py
```

This will create the PCA models from your training data.

### 3. Start the Backend API

In Terminal 1:

```bash
# Using uv
uv run backend_api.py

# Or with python
python backend_api.py
```

You should see:
```
==================================================
🚀 Hand Detection Backend API
==================================================
📍 Running on: http://localhost:5001
📍 Prediction endpoint: http://localhost:5001/predict
📍 Health check: http://localhost:5001/health
==================================================
```

**Note**: The backend runs on port 5001 (not 5000) to avoid conflicts with macOS AirPlay Receiver.

### 4. Open the HTML Frontend

Open `realtime_hand_detector.html` in your browser:

```bash
# On macOS
open realtime_hand_detector.html

# On Linux
xdg-open realtime_hand_detector.html

# Or manually open in Chrome/Firefox
```

### 5. Connect to Sensor

1. In the HTML interface, verify the WebSocket URL points to your sensor device
2. Click the "🔌 Connect" button
3. Your mobile device should start sending accelerometer data
4. Watch real-time predictions appear on the PCA plot

## How It Works

### Data Flow

1. **Sensor → WebSocket**: Mobile device sends accelerometer data (x, y, z) via WebSocket
2. **HTML Frontend**: Receives sensor data and sends it to backend API
3. **Backend API**:
   - Calculates magnitude: `sqrt(x² + y² + z²)`
   - Scales features using trained scaler
   - Makes prediction using Random Forest model
   - Computes PCA coordinates for visualization
   - Returns: `{hand: "left/right", confidence: 0.0-1.0, pc1: X, pc2: Y}`
4. **Visualization**: Plots predictions on 2D PCA space with trajectory tracking

### Features

- **Real-time PCA Visualization**: 2D projection of feature space
- **Trajectory Tracking**: White line connecting predictions chronologically
- **Latest Point Highlight**: Yellow-rimmed marker for most recent prediction
- **Color Coding**: Red stars for left hand, blue stars for right hand
- **Live Statistics**: Total predictions, left/right counts, buffer size
- **Recent Predictions Table**: Last 10 predictions with timestamps
- **Confidence Display**: Large centered box showing current hand prediction
- **Auto-buffering**: Keeps last 100 predictions for performance

## Troubleshooting

### Backend API Issues

**Error: "ModuleNotFoundError: No module named 'flask_cors'"**
```bash
pip install flask-cors
```

**Error: "FileNotFoundError: hand_classifier_model.pkl"**
- Make sure you ran the `which_hand_you_use.ipynb` notebook first to train and save the models
- Check that all `.pkl` files are in the same directory as `backend_api.py`

**Error: "Address already in use"**
- The backend already runs on port 5001 to avoid conflicts
- If port 5001 is also in use, change the `PORT` variable in [backend_api.py:84](backend_api.py#L84):
  ```python
  PORT = 5002  # Or any available port
  ```
  Then update the `BACKEND_URL` in [realtime_hand_detector.html:381](realtime_hand_detector.html#L381) to match

### Frontend Issues

**Error: "Failed to fetch" in console**
- Make sure the backend API is running on http://localhost:5001
- Check CORS is enabled (should be automatic with flask-cors)
- Verify the backend health endpoint: http://localhost:5001/health

**WebSocket connection fails**
- Check the WebSocket URL matches your sensor device IP address
- Ensure your mobile device is on the same network
- Verify the sensor app is running and broadcasting

**No predictions appearing**
- Open browser console (F12) and check for errors
- Verify backend API is responding: `curl http://localhost:5000/health`
- Check that WebSocket is connected (status badge should show "🟢 Connected")

### Performance Issues

**Predictions are slow**
- The API makes synchronous calls - this is expected
- Each prediction takes ~10-50ms depending on your machine
- Consider reducing sensor data rate if overwhelmed

**Plot is laggy**
- Clear data buffer (click "🗑️ Clear Data")
- The system keeps only last 100 points automatically
- Close other browser tabs

## Testing the System

### Test Backend API

```bash
# Health check
curl http://localhost:5001/health

# Test left hand prediction
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"x": 1.2, "y": -0.5, "z": 9.8}'

# Test right hand prediction
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"x": -2.5, "y": 1.2, "z": 9.5}'
```

Expected response:
```json
{
  "hand": "left",
  "confidence": 0.91,
  "pc1": 2.79,
  "pc2": 0.61
}
```

### Test Without Sensor

You can test the system without a real sensor by modifying the HTML to generate fake data:

```javascript
// Add after line 411 in HTML
setInterval(() => {
    if (isConnected) {
        // Simulate sensor data
        const fakeData = {
            values: [
                Math.random() * 4 - 2,  // x: -2 to 2
                Math.random() * 4 - 2,  // y: -2 to 2
                9.8 + Math.random() - 0.5  // z: around 9.8
            ]
        };
        ws.onmessage({ data: JSON.stringify(fakeData) });
    }
}, 100);  // Every 100ms
```

## Model Performance

From `which_hand_you_use.ipynb`:
- **Accuracy**: ~94.6%
- **Model**: Random Forest Classifier (100 estimators)
- **Features**: X, Y, Z accelerometer + magnitude
- **Training Data**: 2 subjects, both hands

## Next Steps

- Add historical PCA points from training data as background reference
- Implement WebSocket reconnection logic
- Add export functionality for collected predictions
- Create mobile app version
- Add real-time accuracy metrics if ground truth is available
