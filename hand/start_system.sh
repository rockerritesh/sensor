#!/bin/bash

# Real-Time Hand Detection System Launcher
# This script starts the backend API and opens the frontend

echo "=========================================="
echo "🚀 Starting Hand Detection System"
echo "=========================================="
echo ""

# Check if all model files exist
if [ ! -f "hand_classifier_model.pkl" ] || [ ! -f "hand_classifier_scaler.pkl" ]; then
    echo "❌ Error: ML model files not found!"
    echo "Please run the Jupyter notebook 'which_hand_you_use.ipynb' first to train the model."
    exit 1
fi

if [ ! -f "pca_model.pkl" ] || [ ! -f "scaler_pca.pkl" ]; then
    echo "⚠️  PCA model files not found. Generating them now..."
    uv run save_pca_models.py || python save_pca_models.py
    echo ""
fi

echo "✅ All model files ready"
echo ""

# Start backend in background
echo "📡 Starting backend API on port 5001..."
uv run backend_api.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 3

# Check if backend is running
if curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "✅ Backend API is running (PID: $BACKEND_PID)"
else
    echo "❌ Failed to start backend API"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🌐 Opening frontend in browser..."
sleep 1

# Open HTML in default browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open realtime_hand_detector.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open realtime_hand_detector.html
else
    echo "Please manually open: realtime_hand_detector.html"
fi

echo ""
echo "=========================================="
echo "✅ System is ready!"
echo "=========================================="
echo ""
echo "📍 Backend API: http://localhost:5001"
echo "📍 Health Check: http://localhost:5001/health"
echo "📍 Frontend: realtime_hand_detector.html"
echo ""
echo "Next steps:"
echo "1. Click '🔌 Connect' button in the browser"
echo "2. Watch real-time predictions on the PCA plot"
echo ""
echo "To stop the backend:"
echo "  kill $BACKEND_PID"
echo ""
echo "Or press Ctrl+C (this will leave backend running)"
echo "=========================================="

# Keep script running
wait $BACKEND_PID
