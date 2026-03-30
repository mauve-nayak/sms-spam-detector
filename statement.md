# Problem Statement

Spam messages are a major nuisance for users, often containing fraudulent links, scams, or unwanted promotional content. Manually identifying such spam is time-consuming and unreliable. The goal of this project is to automate the process of detecting whether an SMS message is **spam** or **ham (not spam)** using machine learning and text processing techniques.


# Scope of the Project

- The project focuses on classifying **single text messages** as spam or ham.
- It uses machine learning models trained on the popular SMS Spam Collection dataset.
- The project includes:
  - Data loading and preprocessing
  - Model training and evaluation
  - Saving and reusing the trained model
  - A command-line interface for real-time user predictions
- The system works entirely offline once trained.
- The model handles common spam variations, including obfuscated messages and keyword-based scams.

---

# Target Users

- **General users** who want a simple way to detect spam in SMS content.
- **Students and beginners** learning machine learning and NLP concepts.
- **Developers** exploring small-scale text classification projects.
- **Educators** demonstrating TF-IDF and Logistic Regression models.
- **Cybersecurity learners** analyzing spam detection techniques.

---

# High-Level Features

- Automatic spam vs. ham classification.
- Combined **word-level** and **character-level** TF-IDF feature extraction.
- Logistic Regression–based machine learning model.
- Console-based SMS input and prediction output.
- Automatic model loading or training based on availability.
- Displays prediction confidence scores.
- Fully offline processing after initial setup.
- Clean console output on every run.
