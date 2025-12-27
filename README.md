# Sensor Data Research Projects

This repository contains two main research projects for real-time sensor-based detection systems.

## Projects

### 1. Hand Detection ([hand/](hand/))
Real-time hand detection system that identifies which hand (left or right) is holding a mobile device using accelerometer data.

**Key Features:**
- 94.6% accuracy using Random Forest classifier
- Real-time PCA visualization
- Flask REST API + HTML frontend
- WebSocket sensor integration

**Quick Start:**
```bash
cd hand
./start_system.sh
```

See [hand/README.md](hand/README.md) for details.

### 2. Heart Rate Detection ([heart/](heart/))
Real-time heart rate monitoring system using sensor data.

**Key Features:**
- Real-time heart rate detection
- Streamlit web interface
- Live data visualization

**Quick Start:**
```bash
cd heart
streamlit run streamlit_heartrate.py
```

See [heart/README.md](heart/README.md) for details.

## Shared Resources ([shared/](shared/))

Common utilities, training data, and analysis tools used across both projects.

**Contents:**
- Data collection scripts
- Training datasets
- Analysis notebooks
- Visualization plots

See [shared/README.md](shared/README.md) for details.

## Project Structure

```
sensor-data/
├── hand/                      # Hand detection project
│   ├── backend_api.py         # Flask API server
│   ├── realtime_hand_detector.html  # Web frontend
│   ├── which_hand_you_use.ipynb     # Model training notebook
│   ├── *.pkl                  # Trained models
│   ├── start_system.sh        # Launch script
│   └── README.md
│
├── heart/                     # Heart rate detection project
│   ├── realtime_heartrate.py  # Real-time detector
│   ├── streamlit_heartrate.py # Streamlit interface
│   └── README.md
│
├── shared/                    # Shared resources
│   ├── collect_data.py        # Data collection utility
│   ├── analysis.py            # Analysis scripts
│   ├── *.csv                  # Training datasets
│   └── README.md
│
├── pyproject.toml             # Python dependencies
├── uv.lock                    # Locked dependencies
└── README.md                  # This file
```

## Requirements

Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

## Development

Each project is self-contained in its own directory. Navigate to the specific project folder to work on it:

- For hand detection: `cd hand/`
- For heart rate: `cd heart/`
- For shared utilities: `cd shared/`

## Data Collection

Both projects use sensor data from mobile devices. See the shared data collection scripts in [shared/](shared/) for utilities to collect and process sensor readings.

## Contributing

When adding new features:
1. Place project-specific files in the appropriate directory (hand/ or heart/)
2. Place shared utilities in shared/
3. Update the relevant README.md file
4. Keep model files and training data organized

## License

Research project - see individual project directories for specific licensing.
