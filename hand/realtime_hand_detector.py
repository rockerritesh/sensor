import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import websocket
import json
import pickle
import threading
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import deque
import datetime

# Page config
st.set_page_config(
    page_title="Hand Detection - Real-time",
    page_icon="🤚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-left {
        color: #ef4444;
        font-size: 24px;
        font-weight: bold;
    }
    .prediction-right {
        color: #3b82f6;
        font-size: 24px;
        font-weight: bold;
    }
    h1 {
        color: #f1f5f9;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🤚 Real-Time Hand Detection System</h1>", unsafe_allow_html=True)

# Global variables for models and data (accessible from thread)
GLOBAL_MODELS = {
    'ml_model': None,
    'ml_scaler': None,
    'pca_model': None,
    'scaler_pca': None
}

GLOBAL_DATA = {
    'realtime_data': deque(maxlen=100),
    'current_prediction': None,
    'confidence': 0.0,
    'left_count': 0,
    'right_count': 0
}

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
    st.session_state.ws_connected = False
    st.session_state.ws = None

@st.cache_data
def load_historical_data():
    """Load and combine all historical data"""
    s1_left = pd.read_csv('hand_data/accelerometer/s-1_left_hand.csv')
    s1_right = pd.read_csv('hand_data/accelerometer/s-1_right_hand.csv')
    s2_left = pd.read_csv('hand_data/accelerometer/s-2_left_hand.csv')
    s2_right = pd.read_csv('hand_data/accelerometer/s-2_right_hand.csv')

    s1_left['hand'] = 'left'
    s1_right['hand'] = 'right'
    s2_left['hand'] = 'left'
    s2_right['hand'] = 'right'

    all_data = pd.concat([s1_left, s1_right, s2_left, s2_right], ignore_index=True)
    all_data['magnitude'] = np.sqrt(all_data['x']**2 + all_data['y']**2 + all_data['z']**2)

    return all_data

@st.cache_resource
def create_pca_model(data):
    """Create and fit PCA model on historical data"""
    X = data[['x', 'y', 'z', 'magnitude']].values

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return pca, scaler, X_pca

@st.cache_resource
def load_ml_model():
    """Load the trained ML model and scaler"""
    try:
        with open('hand_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('hand_classifier_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the training notebook first.")
        return None, None

def create_pca_plot(data, X_pca, realtime_points=None):
    """Create interactive PCA scatter plot"""
    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'hand': data['hand'].values
    })

    # Sample data for better performance (plot 5000 points max)
    if len(pca_df) > 5000:
        pca_df = pca_df.sample(n=5000, random_state=42)

    fig = go.Figure()

    # Plot historical data
    for hand, color in [('left', '#ef4444'), ('right', '#3b82f6')]:
        hand_data = pca_df[pca_df['hand'] == hand]
        fig.add_trace(go.Scatter(
            x=hand_data['PC1'],
            y=hand_data['PC2'],
            mode='markers',
            name=f'{hand.capitalize()} Hand (Historical)',
            marker=dict(
                size=6,
                color=color,
                opacity=0.3,
                line=dict(width=0)
            ),
            hovertemplate=f'<b>{hand.capitalize()} Hand</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
        ))

    # Plot real-time points with trajectory
    if realtime_points and len(realtime_points) > 0:
        rt_df = pd.DataFrame(realtime_points)
        if not rt_df.empty:
            # Add trajectory line for all points (chronological order)
            fig.add_trace(go.Scatter(
                x=rt_df['pc1'],
                y=rt_df['pc2'],
                mode='lines',
                name='Trajectory',
                line=dict(
                    color='white',
                    width=2,
                    dash='solid'
                ),
                opacity=0.6,
                hoverinfo='skip',
                showlegend=True
            ))

            # Add markers for each hand type
            for hand, color, symbol in [('left', '#ef4444', 'star'), ('right', '#3b82f6', 'star')]:
                hand_rt = rt_df[rt_df['prediction'] == hand]
                if not hand_rt.empty:
                    fig.add_trace(go.Scatter(
                        x=hand_rt['pc1'],
                        y=hand_rt['pc2'],
                        mode='markers',
                        name=f'{hand.capitalize()} Hand (Real-time)',
                        marker=dict(
                            size=15,
                            color=color,
                            symbol=symbol,
                            line=dict(width=2, color='white'),
                            opacity=0.9
                        ),
                        hovertemplate=f'<b>REAL-TIME: {hand.capitalize()}</b><br>Confidence: %{{customdata:.1%}}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>',
                        customdata=hand_rt['confidence']
                    ))

            # Highlight the most recent point
            if len(rt_df) > 0:
                latest = rt_df.iloc[-1]
                latest_color = '#ef4444' if latest['prediction'] == 'left' else '#3b82f6'
                fig.add_trace(go.Scatter(
                    x=[latest['pc1']],
                    y=[latest['pc2']],
                    mode='markers',
                    name='Latest Point',
                    marker=dict(
                        size=25,
                        color=latest_color,
                        symbol='circle',
                        line=dict(width=4, color='yellow'),
                        opacity=1.0
                    ),
                    hovertemplate=f'<b>LATEST: {latest["prediction"].capitalize()}</b><br>Confidence: {latest["confidence"]:.1%}<br>PC1: {latest["pc1"]:.2f}<br>PC2: {latest["pc2"]:.2f}<extra></extra>',
                    showlegend=True
                ))

    fig.update_layout(
        title={
            'text': '🎯 PCA Feature Space - Hand Detection',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#f1f5f9'}
        },
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        plot_bgcolor='#1f2937',
        paper_bgcolor='#0e1117',
        font=dict(color='#f1f5f9'),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(31, 41, 55, 0.8)',
            bordercolor='#4b5563',
            borderwidth=1
        ),
        height=600
    )

    return fig

def predict_hand(x, y, z, model, scaler, pca, pca_scaler):
    """Predict hand from sensor data"""
    if model is None or scaler is None:
        return None, 0.0, None, None

    # Calculate magnitude
    magnitude = np.sqrt(x**2 + y**2 + z**2)

    # Create feature array
    features = np.array([[x, y, z, magnitude]])

    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    hand = 'right' if prediction == 1 else 'left'
    confidence = probability[prediction]

    # Get PCA coordinates
    pca_features = pca_scaler.transform(features)
    pca_coords = pca.transform(pca_features)
    pc1, pc2 = pca_coords[0]

    return hand, confidence, pc1, pc2

def on_message(ws, message):
    """WebSocket message handler"""
    try:
        values = json.loads(message)['values']
        x, y, z = values[0], values[1], values[2]
        timestamp = datetime.datetime.now()

        # Make prediction using global models
        hand, confidence, pc1, pc2 = predict_hand(
            x, y, z,
            GLOBAL_MODELS['ml_model'],
            GLOBAL_MODELS['ml_scaler'],
            GLOBAL_MODELS['pca_model'],
            GLOBAL_MODELS['scaler_pca']
        )

        if hand:
            # Store real-time data in global variable
            GLOBAL_DATA['realtime_data'].append({
                'timestamp': timestamp,
                'x': x,
                'y': y,
                'z': z,
                'prediction': hand,
                'confidence': confidence,
                'pc1': pc1,
                'pc2': pc2
            })

            # Update current prediction
            GLOBAL_DATA['current_prediction'] = hand
            GLOBAL_DATA['confidence'] = confidence

            # Update counts
            if hand == 'left':
                GLOBAL_DATA['left_count'] += 1
            else:
                GLOBAL_DATA['right_count'] += 1

            # Debug print
            print(f"✅ Prediction: {hand} ({confidence:.1%}) - Total: {GLOBAL_DATA['left_count'] + GLOBAL_DATA['right_count']}")

    except Exception as e:
        print(f"Error processing message: {e}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_code, reason):
    st.session_state.ws_connected = False
    print("WebSocket connection closed")

def on_open(ws):
    st.session_state.ws_connected = True
    print("✅ Connected to sensor!")

def connect_websocket(url):
    """Connect to WebSocket in a separate thread"""
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    st.session_state.ws = ws
    ws.run_forever()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # WebSocket URL input
    ws_url = st.text_input(
        "WebSocket URL",
        value="ws://192.168.1.70:8081/sensor/connect?type=android.sensor.accelerometer",
        help="Enter your sensor WebSocket URL"
    )

    # Connect/Disconnect button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔌 Connect", disabled=st.session_state.ws_connected):
            # Start WebSocket in background thread
            ws_thread = threading.Thread(target=connect_websocket, args=(ws_url,), daemon=True)
            ws_thread.start()
            with st.spinner("Connecting..."):
                time.sleep(1)  # Wait for connection
            st.success("Connected!")

    with col2:
        if st.button("⏹️ Disconnect", disabled=not st.session_state.ws_connected):
            if st.session_state.ws:
                st.session_state.ws.close()
                st.session_state.ws_connected = False
                st.info("Disconnected")

    # Connection status
    st.markdown("### Connection Status")
    if st.session_state.ws_connected:
        st.markdown("🟢 **Connected**")
    else:
        st.markdown("�� **Disconnected**")

    st.markdown("---")

    # Statistics
    st.markdown("## 📊 Statistics")
    total = GLOBAL_DATA['left_count'] + GLOBAL_DATA['right_count']
    st.metric("Total Predictions", total, delta=f"+{len(GLOBAL_DATA['realtime_data'])} in buffer")
    st.metric("Left Hand", GLOBAL_DATA['left_count'])
    st.metric("Right Hand", GLOBAL_DATA['right_count'])

    st.markdown("---")

    # Clear data button
    if st.button("🗑️ Clear Real-time Data"):
        GLOBAL_DATA['realtime_data'].clear()
        GLOBAL_DATA['left_count'] = 0
        GLOBAL_DATA['right_count'] = 0
        GLOBAL_DATA['current_prediction'] = None
        GLOBAL_DATA['confidence'] = 0.0
        st.rerun()

# Main content
# Load data and models
with st.spinner("Loading historical data and models..."):
    st.session_state.historical_data = load_historical_data()
    pca_model, scaler_pca, X_pca = create_pca_model(st.session_state.historical_data)
    ml_model, ml_scaler = load_ml_model()

    # Store in global variables for thread access
    GLOBAL_MODELS['pca_model'] = pca_model
    GLOBAL_MODELS['scaler_pca'] = scaler_pca
    GLOBAL_MODELS['ml_model'] = ml_model
    GLOBAL_MODELS['ml_scaler'] = ml_scaler

# Status info
if st.session_state.ws_connected:
    if len(GLOBAL_DATA['realtime_data']) == 0:
        st.info("🔄 Connected - Waiting for sensor data...")
    else:
        st.success(f"✅ Receiving data - {len(GLOBAL_DATA['realtime_data'])} predictions in buffer")

# Current prediction display
if GLOBAL_DATA['current_prediction']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pred_class = "prediction-left" if GLOBAL_DATA['current_prediction'] == "left" else "prediction-right"
        emoji = "👈" if GLOBAL_DATA['current_prediction'] == "left" else "👉"
        st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #1f2937; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='color: #f1f5f9;'>Current Prediction</h2>
                <p class='{pred_class}'>{emoji} {GLOBAL_DATA['current_prediction'].upper()} HAND</p>
                <p style='color: #9ca3af; font-size: 18px;'>Confidence: {GLOBAL_DATA['confidence']:.1%}</p>
            </div>
        """, unsafe_allow_html=True)

# PCA Plot
st.markdown("### 🎯 Real-time Feature Space Visualization")
pca_plot = create_pca_plot(
    st.session_state.historical_data,
    X_pca,
    list(GLOBAL_DATA['realtime_data'])
)
st.plotly_chart(pca_plot, use_container_width=True)

# Real-time data table
if GLOBAL_DATA['realtime_data']:
    st.markdown("### 📋 Recent Predictions")

    # Convert to DataFrame
    rt_df = pd.DataFrame(list(GLOBAL_DATA['realtime_data']))
    rt_df = rt_df.sort_values('timestamp', ascending=False).head(10)

    # Format for display
    display_df = rt_df[['timestamp', 'x', 'y', 'z', 'prediction', 'confidence']].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S.%f').str[:-3]
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    display_df.columns = ['Time', 'X', 'Y', 'Z', 'Prediction', 'Confidence']

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# Time series plot
if len(GLOBAL_DATA['realtime_data']) > 1:
    st.markdown("### 📈 Sensor Data Over Time")

    rt_df = pd.DataFrame(list(GLOBAL_DATA['realtime_data']))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('X-Axis', 'Y-Axis', 'Z-Axis', 'Confidence'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # X, Y, Z plots
    for idx, (axis, row, col) in enumerate([('x', 1, 1), ('y', 1, 2), ('z', 2, 1)]):
        for hand, color in [('left', '#ef4444'), ('right', '#3b82f6')]:
            hand_data = rt_df[rt_df['prediction'] == hand]
            if not hand_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=hand_data['timestamp'],
                        y=hand_data[axis],
                        mode='lines+markers',
                        name=f'{hand.capitalize()}',
                        line=dict(color=color),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )

    # Confidence plot
    for hand, color in [('left', '#ef4444'), ('right', '#3b82f6')]:
        hand_data = rt_df[rt_df['prediction'] == hand]
        if not hand_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=hand_data['timestamp'],
                    y=hand_data['confidence'],
                    mode='lines+markers',
                    name=f'{hand.capitalize()}',
                    line=dict(color=color),
                    showlegend=False
                ),
                row=2, col=2
            )

    fig.update_layout(
        height=500,
        plot_bgcolor='#1f2937',
        paper_bgcolor='#0e1117',
        font=dict(color='#f1f5f9'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# Auto-refresh when connected
if st.session_state.ws_connected:
    # Add a small delay and rerun to show updates
    time.sleep(0.5)
    st.rerun()
