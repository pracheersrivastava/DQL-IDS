# 🛡️ Deep Q-Learning Intrusion Detection System (IDS)

An AI-based lightweight Intrusion Detection System trained on the NSL-KDD dataset using **Deep Q-Learning (DQL)**. This IDS is capable of classifying and detecting multiple cyberattacks such as DoS, Probe, U2R, R2L, and more, with future support for both **offline log detection** and **real-time monitoring**.

---

### 🚀 Project Overview

This is a semester-long academic project aimed at detecting cyberattacks using Reinforcement Learning. The dataset used is **NSL-KDD** and the agent learns to classify attacks into 5 major classes: `Normal`, `DoS`, `Probe`, `U2R`, and `R2L`.

The model follows the **Deep Q-Learning (DQL)** workflow and incorporates:
- A custom reward system
- Experience replay
- Epsilon-Greedy action selection
- Q-value updates via a neural network

---

### 🧱 Project Structure

```
deep-qlearning-ids/
├── data/               → Original NSL-KDD files (KDDTrain+.TXT, KDDTest+.TXT)
├── output/             → Cleaned data, trained models (.keras)
├── notebooks/          → Scripts for preprocessing, DQL model, training
├── .gitignore          → Python cache/model ignore list
└── README.md           → You’re here
```

---

### 📦 Workflow Completed

✅ **Data Preprocessing**  
Loaded `KDDTrain+.TXT`, assigned column names, cleaned dataset, grouped raw attack labels into 5 major categories, and encoded categorical features.

✅ **Model Architecture**  
Designed a Deep Q-Learning model using Keras with:
- Input layer of 41 features
- Hidden layers: Dense(128) → Dense(64)
- Output layer: 5 neurons (one for each attack class)

✅ **Training**  
Implemented:
- Epsilon-Greedy exploration
- Experience Replay (buffer size: 2000)
- Reward = `+1` for correct prediction, `-1` otherwise
- Trained on 100k+ samples using small batches

🔁 Training Sample Log:
```
🚀 Starting Training...
    Processing sample 0/100000...
    Processing sample 5000/100000...
✅ Episode 1/30 - Loss: 0.2456 - Epsilon: 0.985
```

🧠 Trained model saved to: `C:/Users/Pracheer/Desktop/DQL IDS/output/dql_model_trained.keras`

---

### 🧠 Technologies Used

- Python 3.12
- Pandas, NumPy, Scikit-learn
- TensorFlow / Keras
- NSL-KDD Dataset
- Matplotlib, Seaborn (for evaluation — coming next)

---

### 📊 Dataset Used: NSL-KDD

> From the [NSL-KDD-Dataset](https://github.com/jmnwong/NSL-KDD-Dataset)

- 41 features + attack labels
- Grouped into 5 major classes:
  - 🟢 `Normal`
  - 🔴 `DoS`
  - 🟡 `Probe`
  - 🔵 `U2R`
  - 🟠 `R2L`

---

### 🛠️ Current Files

| Script                    | Purpose                              |
|---------------------------|--------------------------------------|
| `data_preprocessing.py`   | Clean and encode KDDTrain+.TXT       |
| `dql_model.py`            | Build and compile the model          |
| `dql_training.py`         | Train the model with RL logic        |
| *(coming soon)*           | Evaluation, confusion matrix, graphs |

---

### 📍 Next Steps (Review 2 → 3)

- ⏳ Evaluate model: Accuracy, F1-score, Confusion Matrix  
- ⏳ Create offline & real-time attack detection interface  
- ⏳ GUI with log upload or live sniffing  
- ⏳ Package into `.exe` or web-based dashboard  
- ⏳ Optional: Deploy on LAN / Cloud (Heroku/Firebase)

---

### 👤 Author

> Developed by **Pracheer Srivastava, Saurav Chourasia, Sarthak Singh, Subeer Srivastava, Suchanda Dutta**  
B.Tech CSE (Cyber Security & Digital Forensics)  
VIT Bhopal University – April 2025

## 📘 License
This is a university academic project. For educational and non-commercial use only.

---
