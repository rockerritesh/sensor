# Hand Detection Project

Real-time hand detection system using accelerometer data from mobile sensors. Determines which hand (left or right) is holding the device using machine learning.

## Features

- Real-time ML predictions using Random Forest classifier (94.6% accuracy)
- PCA visualization with 2D projection (79% variance explained)
- WebSocket integration for live sensor data
- Flask REST API backend
- Interactive HTML/JavaScript frontend
- Trajectory tracking and confidence scoring

## Quick Start

### Option 1: Use the Launch Script
```bash
./start_system.sh
```

### Option 2: Manual Start
```bash
# Start backend API
uv run backend_api.py

# Open frontend in browser
open realtime_hand_detector.html
```

## Project Files

### Core Files
- `backend_api.py` - Flask REST API serving ML predictions
- `realtime_hand_detector.html` - HTML/JavaScript frontend with visualization
- `realtime_hand_detector.py` - Alternative Python-based detector
- `start_system.sh` - Automated system launcher script

### Model Files
- `hand_classifier_model.pkl` - Trained Random Forest classifier (75 MB)
- `hand_classifier_scaler.pkl` - Feature scaler for ML model
- `pca_model.pkl` - PCA model for dimensionality reduction
- `scaler_pca.pkl` - Scaler for PCA features

### Training & Analysis
- `which_hand_you_use.ipynb` - Training notebook (model development)
- `save_pca_models.py` - Script to generate PCA models
- `hand_data/` - Training data directory

### Documentation
- `QUICK_START.md` - Quick start guide
- `REALTIME_SETUP.md` - Detailed setup instructions

## Architecture

```
Mobile Sensor → WebSocket → HTML Frontend → Flask API → ML Model → Real-time PCA Visualization
```

## How It Works

1. Mobile device sends accelerometer data (x, y, z) via WebSocket
2. Frontend receives data and sends to Flask backend API
3. Backend calculates magnitude: √(x² + y² + z²)
4. Features are scaled using trained scaler
5. Random Forest model predicts hand (left/right)
6. PCA coordinates computed for visualization
7. Results displayed on interactive 2D plot with trajectory

## Model Performance

- **Accuracy**: 94.6%
- **Model**: Random Forest (100 estimators)
- **Features**: x, y, z accelerometer + magnitude
- **Training Data**: 149,762 samples (2 subjects)
- **PCA Variance**: 79.03% (PC1: 54.7%, PC2: 24.4%)

## Requirements

- Flask
- Flask-CORS
- scikit-learn
- NumPy
- pickle

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Get hand prediction
  - Request: `{"x": float, "y": float, "z": float}`
  - Response: `{"hand": "left/right", "confidence": float, "pc1": float, "pc2": float}`

## Troubleshooting

See [REALTIME_SETUP.md](REALTIME_SETUP.md) for detailed troubleshooting steps.

## Next Steps

- Collect more training data from additional subjects
- Add historical PCA points as background reference
- Implement WebSocket reconnection logic
- Export prediction data to CSV
- Create mobile app version
