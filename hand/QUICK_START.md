# Quick Start Guide

## Real-Time Hand Detection System - Ready to Use!

Your real-time hand detection system is now fully operational with actual ML model predictions.

## What's New

The system now uses your trained Random Forest model (94.6% accuracy) instead of placeholder predictions.

### Architecture

```
Mobile Sensor → WebSocket → HTML Frontend → Flask API → ML Model → Real-time PCA Visualization
```

## Files Created

1. **[backend_api.py](backend_api.py)** - Flask REST API serving ML predictions
2. **[realtime_hand_detector.html](realtime_hand_detector.html)** - Updated to use backend API
3. **[save_pca_models.py](save_pca_models.py)** - Script to generate PCA models
4. **[REALTIME_SETUP.md](REALTIME_SETUP.md)** - Complete setup documentation

## Quick Start (3 Steps)

### Step 1: Start the Backend

```bash
uv run backend_api.py
```

Output:
```
==================================================
🚀 Hand Detection Backend API
==================================================
📍 Running on: http://localhost:5001
📍 Prediction endpoint: http://localhost:5001/predict
📍 Health check: http://localhost:5001/health
==================================================
```

### Step 2: Open the Frontend

```bash
open realtime_hand_detector.html
```

### Step 3: Connect to Your Sensor

1. Click "🔌 Connect" button in the HTML interface
2. Watch real-time predictions with 91-100% confidence!

## System Status

✅ Backend API running on port 5001
✅ Flask-CORS installed and configured
✅ All model files generated:
- `hand_classifier_model.pkl` (75 MB)
- `hand_classifier_scaler.pkl` (512 B)
- `pca_model.pkl` (870 B)
- `scaler_pca.pkl` (512 B)

## Test the Backend

```bash
# Quick health check
curl http://localhost:5001/health

# Test prediction
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"x": 1.2, "y": -0.5, "z": 9.8}'
```

Expected:
```json
{
  "hand": "left",
  "confidence": 0.91,
  "pc1": 2.79,
  "pc2": 0.61
}
```

## Features

- **Real ML Predictions**: Using your trained Random Forest model
- **High Accuracy**: 91-100% confidence on predictions
- **PCA Visualization**: 2D projection (79% variance explained)
- **Trajectory Tracking**: White line connecting predictions
- **Color Coding**: Red stars (left), blue stars (right)
- **Live Statistics**: Counts, confidence, buffer size
- **Auto-buffering**: Keeps last 100 predictions

## What the System Does

1. **Receives** accelerometer data (x, y, z) from mobile sensor
2. **Calculates** magnitude: √(x² + y² + z²)
3. **Scales** features using trained scaler
4. **Predicts** hand using Random Forest (100 trees)
5. **Computes** PCA coordinates for visualization
6. **Displays** prediction with confidence on interactive plot

## Troubleshooting

**Backend not starting?**
- Port 5001 might be in use
- Check with: `lsof -i :5001`

**Frontend not connecting?**
- Verify backend is running: `curl http://localhost:5001/health`
- Check browser console (F12) for errors

**No sensor data?**
- Update WebSocket URL in HTML to match your device IP
- Ensure device is on same network

## Model Performance

From training ([which_hand_you_use.ipynb](which_hand_you_use.ipynb)):
- **Accuracy**: 94.6%
- **Model**: Random Forest (100 estimators)
- **Features**: x, y, z, magnitude
- **Training Data**: 149,762 samples (2 subjects)
- **PCA Variance**: 79.03% (PC1: 54.7%, PC2: 24.4%)

## Next Steps

1. **Collect More Data**: Train on more subjects for better generalization
2. **Add Historical Points**: Show training data as background on PCA plot
3. **Export Predictions**: Add CSV download functionality
4. **Real-time Metrics**: Show rolling accuracy if ground truth available
5. **Mobile App**: Create native mobile version

---

For detailed documentation, see [REALTIME_SETUP.md](REALTIME_SETUP.md)
