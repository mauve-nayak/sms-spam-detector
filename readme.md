# 📱 SMS Spam Detector

A machine-learning based SMS classifier that predicts whether a given text message is **spam** or **ham (not spam)**.  
It uses TF-IDF vectorization and a Logistic Regression model with additional character-level features for high spam detection accuracy.

---

## ARJUN AGNIHOTRI

```bash
https://www.arjunagnihotri.com/
```

## 📌 Overview

This project provides an end-to-end SMS Spam Detection system using Python and scikit-learn. It:

- Loads and cleans the SMS Spam dataset
- Trains a machine learning model (TF-IDF + Logistic Regression)
- Saves the trained model for reuse
- Provides a clean console interface for prediction
- Displays prediction confidence
- Clears the console automatically on each execution

---

## ✨ Features

- ✔️ Detects spam vs. ham with high accuracy
- ✔️ Automatically loads or trains the model
- ✔️ Uses both **word-level** and **character-level** TF-IDF
- ✔️ Simple and clean CLI interface
- ✔️ Offline — no internet required
- ✔️ Lightweight and fast

---

## 🛠️ Technologies / Tools Used

- **Python 3.x**
- **Pandas**
- **scikit-learn**
- **Joblib**
- **Windows CMD / PowerShell**

---

## 📥 Installation & Setup

### 1️⃣ Clone or download the repository

```bash
git clone <your-repo-link>
cd sms-spam-detector
```

### 2️⃣ Install dependencies

```bash
sklearn-env\Scripts\activate
pip install pandas scikit-learn joblib
```

### 3️⃣ Download the dataset

```bash
Place the spam.csv file (from Kaggle) inside your project folder.
```

### 4️⃣ Ensure project structure

```bash
SMS Spam Detector/
├── main.py
├── spam.csv
└── README.md
```

### ▶️ Running the Project

```bash
python main.py
```

Each run will:

Clear the console

Display a short program description

Ask for an SMS input

Load or train the model

Predict spam/ham with confidence score

### Example:

```bash
Enter SMS message to classify:
> Congratulations! You've won a prize!
Prediction: spam (confidence: 98.1%)

```

## 🧪 Testing Instructions

#### Try messages of different types:

```bash
✔️ Normal (ham) messages:

"Hey, are we meeting today?"

"Call me when you're free."

"Don't forget the meeting at 3PM."
```

```bash
✔️ Spam-like messages:

"WIN a FREE 🎁 iPhone! Click here!"

"You have been selected for a $1000 cash prize!"

"Claim your reward now!!!"

The classifier should correctly identify them.
```
