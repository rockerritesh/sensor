"""
Generate and save PCA models from training data
Run this script to create pca_model.pkl and scaler_pca.pkl
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load training data
print("Loading training data...")
s1_left = pd.read_csv('hand_data/accelerometer/s-1_left_hand.csv')
s1_right = pd.read_csv('hand_data/accelerometer/s-1_right_hand.csv')
s2_left = pd.read_csv('hand_data/accelerometer/s-2_left_hand.csv')
s2_right = pd.read_csv('hand_data/accelerometer/s-2_right_hand.csv')

# Add labels
s1_left['hand'] = 'left'
s1_right['hand'] = 'right'
s2_left['hand'] = 'left'
s2_right['hand'] = 'right'

# Combine all data
df = pd.concat([s1_left, s1_right, s2_left, s2_right], ignore_index=True)

print(f"Total samples: {len(df)}")

# Calculate magnitude
df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

# Prepare features for PCA (same as used for visualization)
features_pca = df[['x', 'y', 'z', 'magnitude']].values

# Scale features before PCA
print("Fitting scaler for PCA...")
scaler_pca = StandardScaler()
features_pca_scaled = scaler_pca.fit_transform(features_pca)

# Fit PCA
print("Fitting PCA model...")
pca_model = PCA(n_components=2)
pca_coords = pca_model.fit_transform(features_pca_scaled)

print(f"PCA explained variance ratio: {pca_model.explained_variance_ratio_}")
print(f"Total variance explained: {pca_model.explained_variance_ratio_.sum():.2%}")

# Save models
print("\nSaving models...")
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca_model, f)
print("✅ Saved pca_model.pkl")

with open('scaler_pca.pkl', 'wb') as f:
    pickle.dump(scaler_pca, f)
print("✅ Saved scaler_pca.pkl")

print("\n" + "="*50)
print("🎉 PCA models saved successfully!")
print("="*50)
print("You can now run backend_api.py")
