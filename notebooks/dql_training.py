import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import deque
import random

# === Load Cleaned Data ===
df = pd.read_csv('C:/Users/Pracheer/Desktop/DQL IDS/output/KDDTrain+_cleaned.csv')

X = df.drop(columns=['label', 'difficulty_level', 'attack_class'])
y = df['attack_class']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === Load model from Person 2 ===
from tensorflow.keras.models import load_model
model = load_model('C:/Users/Pracheer/Desktop/DQL IDS/output/dql_model_architecture.keras')

# === DQL Training Config ===
gamma = 0.95                 # Discount factor
epsilon = 1.0                # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
episodes = 30

# === Experience Replay Buffer ===
memory = deque(maxlen=2000)

# === Convert labels to one-hot targets (for reward matching) ===
n_classes = len(np.unique(y_train))

# === Training Loop ===
print("🚀 Starting Training...")
for episode in range(episodes):
    total_loss = 0
    for i in range(len(X_train)):
        if i % 5000 == 0:
            print(f"Processing sample {i}/{len(X_train)}...")
        state = np.array([X_train.iloc[i].values.astype(np.float32)])
        action = y_train[i]                  # True label → use as target

        # --- Predict Q-values ---
        q_values = model.predict(state, verbose=0)[0]

        # --- Choose action (Epsilon-Greedy) ---
        if np.random.rand() <= epsilon:
            predicted_action = np.random.randint(n_classes)
        else:
            predicted_action = np.argmax(q_values)

        # --- Reward Logic ---
        reward = 1 if predicted_action == action else -1

        # --- Predict Q-value for next state (not needed here as we're doing supervised-style DQL) ---
        target = q_values
        target[action] = reward  # Modify the Q-value for the correct class

        # --- Save to Replay Buffer ---
        memory.append((state[0], target))

        # --- Train from Replay Buffer if enough samples ---
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            X_batch = np.array([s for s, _ in minibatch])
            y_batch = np.array([t for _, t in minibatch])
            loss = model.train_on_batch(X_batch, y_batch)
            total_loss += loss

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"✅ Episode {episode+1}/{episodes} - Loss: {total_loss:.4f} - Epsilon: {epsilon:.4f}")

# === Save the trained model ===
model.save('C:/Users/Pracheer/Desktop/DQL IDS/output/dql_model_trained.keras')
print("🧠 Training complete. Model saved.")
