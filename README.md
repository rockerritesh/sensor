# Hand Detection System with Temporal Smoothing

Real-time hand detection system that identifies which hand (left or right) is holding a mobile device using accelerometer data. Features production-ready temporal smoothing to prevent rapid switching between predictions.

## Features

- **High Accuracy**: 94.6% using Random Forest classifier
- **Temporal Smoothing**: 90% reduction in prediction jitter
- **Real-time Visualization**: Interactive 2D/3D PCA plots
- **Multiple Modes**: Majority vote, confidence-weighted, windowed features
- **Production Ready**: Flask REST API + WebSocket integration
- **Easy Setup**: One-command startup script

## Quick Start

```bash
cd hand
./start_system.sh
```

This will:
1. Start the backend API on port 5001
2. Open the frontend in your browser
3. Enable temporal smoothing automatically

See [hand/README.md](hand/README.md) for detailed documentation.

## Key Improvements

### Temporal Smoothing
- **20-sample sliding window** buffer
- **Majority vote** smoothing (default)
- **Confidence-weighted** voting option
- **Windowed features** model (23 statistical features)

### Performance
- Accuracy: 94.6% → 95-96% with smoothing
- Jitter reduction: ~90%
- Warm-up time: 1-2 seconds
- Latency: 15-30ms per prediction

## Project Structure

```
hand/
├── backend_api.py              # Flask API with temporal smoothing
├── realtime_hand_detector.html # Interactive 2D/3D visualization
├── which_hand_you_use.ipynb    # Model training + windowing
├── start_system.sh             # One-command launcher
├── Models:
│   ├── hand_classifier_model.pkl         # Random Forest (94.6%)
│   ├── hand_classifier_windowed_model.pkl # Windowed RF (96%+)
│   ├── pca_model.pkl                      # PCA for visualization
│   └── window_params.pkl                  # Window configuration
├── Documentation:
│   ├── README.md               # Project overview
│   ├── QUICK_START.md          # Quick start guide
│   ├── REALTIME_SETUP.md       # Detailed setup
│   ├── TEMPORAL_SMOOTHING.md   # Technical guide
│   └── WINDOWING_SUMMARY.md    # Implementation summary
└── Scripts:
    ├── save_pca_models.py      # Generate PCA models
    └── realtime_hand_detector.py # Alternative detector
```

## Requirements

Install dependencies using uv:
```bash
uv sync
```

Key dependencies:
- Flask + Flask-CORS (backend API)
- scikit-learn (ML models)
- numpy, scipy (data processing)
- Jupyter (training notebook)

## Documentation

- **[QUICK_START.md](hand/QUICK_START.md)** - Get started in 3 steps
- **[TEMPORAL_SMOOTHING.md](hand/TEMPORAL_SMOOTHING.md)** - Technical deep dive
- **[WINDOWING_SUMMARY.md](hand/WINDOWING_SUMMARY.md)** - Implementation details
- **[REALTIME_SETUP.md](hand/REALTIME_SETUP.md)** - Detailed setup guide

## API Endpoints

The backend provides several endpoints:

- `POST /predict` - Get hand prediction (smoothed)
- `GET /health` - Check API status
- `POST /clear_buffer` - Clear prediction buffer
- `GET /predict_modes` - List available smoothing modes

## Training

To retrain or customize models:

```bash
cd hand
jupyter notebook which_hand_you_use.ipynb
# Run all cells including windowing cells at the end
```

## How It Works

1. **Sensor Data** → WebSocket → Frontend
2. **Frontend** → HTTP POST → Backend API
3. **Backend** → Sliding Window Buffer (20 samples)
4. **Smoothing** → Majority Vote / Confidence / Windowed
5. **Prediction** → Stable hand detection
6. **Visualization** → 2D/3D PCA plot with trajectory

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Update documentation
5. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) for details.

Copyright (c) 2025 Sumit Yadav

## Acknowledgments

Built with temporal smoothing inspired by LSTM concepts for production-ready hand detection.

🚀 Enhanced with [Claude Code](https://claude.com/claude-code)
