# 🤖 Python-Code-Origin

A machine learning-based web application that classifies Python code as Human-Written, AI-Generated, or Hybrid (AI-Refined) using stylometric analysis, transformer embeddings, and feature fusion techniques.

# 📌 Overview

Traditional AI code detectors mainly rely on binary classification, which often misclassifies optimized human-written code as AI-generated. This project introduces a multi-class classification approach that categorizes Python code into Human, AI, and Hybrid classes to better represent real-world AI-assisted coding practices.

The system combines stylometric programming features, CodeBERT embeddings, TF-IDF vectorization, and machine learning models such as Support Vector Machine (SVM) and Random Forest (RF). A Streamlit interface was also developed to provide interactive predictions and feature importance visualizations for improved transparency and usability.

# 🗂️ Dataset Used
https://huggingface.co/datasets/project-droid/DroidCollection
* Only Human-Written, AI-Generated, and AI-Refined Python codes are used

# 💫 Models Used

- SVM (Stylometric)
- Random Forest (Stylometric)
- SVM + CodeBERT
- SVM + Fused Features
- Random Forest + Fused Features

# 🔗 Demo

https://phyton-code-classifier.streamlit.app/
