import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# === Step 1: Load Preprocessed Data ===
df = pd.read_csv('C:/Users/Pracheer/Desktop/DQL IDS/output/KDDTrain+_cleaned.csv')

# === Step 2: Prepare Input and Output ===
X = df.drop(columns=['label', 'difficulty_level', 'attack_class'])
y = df['attack_class']

# Encode attack labels to numbers (e.g., Normal → 0, DoS → 1, ...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === Step 3: Train-Test Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === Step 4: Define the DQL Model ===
def build_dqn_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))  # Q-values for each action
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# === Model Setup ===
input_dim = X_train.shape[1]      # 41 features
output_dim = len(np.unique(y))    # 5 classes: Normal, DoS, Probe, U2R, R2L

model = build_dqn_model(input_dim, output_dim)

print("✅ DQL model built!")
print(model.summary())
model.save('C:/Users/Pracheer/Desktop/DQL IDS/output/dql_model_architecture.keras')
print("Model saved in output folder")   