📊 Multi-Label Text Classification for Toxic Comment Detection
A clean and modular machine learning pipeline for multi-label text classification using Natural Language Processing (NLP) techniques. This project focuses on identifying various types of toxicity in online comments using classical machine learning models.

📑 Table of Contents
Project Overview

Features

Technologies Used

Dataset

Installation

Usage

Results

Visualization

📄 Project Overview

This repository implements a machine learning pipeline for multi-label text classification on toxic comments. It uses TF-IDF vectorization for feature extraction and employs Logistic Regression and Multinomial Naive Bayes models in a One-vs-Rest (OvR) setup. The model can predict multiple toxicity labels for each comment.

🚀 Features

End-to-end text preprocessing:

     Lowercasing

     Regex-based cleaning

     Stopword removal

     Word stemming

Feature extraction using TF-IDF (unigrams & bigrams)

Multi-label classification with:

     Logistic Regression (OvR)

     Multinomial Naive Bayes (OvR)

Performance evaluation:

     Classification reports for each label

     ROC curve visualization for each class

     Clean and reproducible pipeline structure


🛠️ Technologies Used

      Python

      Pandas

      NumPy

    Scikit-learn

      NLTK

    Matplotlib

     Seaborn

📁 Dataset

This project uses the Jigsaw Toxic Comment Classification dataset from Kaggle.

📝 Usage

Place train.csv from Kaggle in the project directory.

Run the main Python script:

    python toxic_comment_classification.py

The pipeline will:

   Clean and preprocess the text

   Train both Logistic Regression and Naive Bayes models

   Evaluate model performance

   Display classification reports and ROC curves for each label

   📊 Results

The classification reports include metrics like precision, recall, and F1-score for each toxicity label.

Example output:
   
   Label: toxic
   
             
              precision    recall  f1-score   support
           0       0.96      0.99      0.97      5735
           1       0.81      0.49      0.61       665
    accuracy                           0.95      6400


📈 Visualization
    
 ROC curves are generated for each label, providing a visual understanding of the tradeoff between the True Positive Rate (TPR) and False Positive Rate (FPR) for the models.
