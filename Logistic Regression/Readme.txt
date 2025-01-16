Multi-Domain Classification with MNIST Data and Language Prediction

Purpose:
This project focuses on implementing classification models for different domains:
1. Binary classification of handwritten digits from the MNIST dataset.
2. Multiclass classification for predicting the language of text samples.
The project demonstrates the use of PCA for dimensionality reduction, logistic regression for classification, 
and TF-IDF for text vectorisation.

---

Key Components:

1. Binary Classification of MNIST Digits:
   - Goal: Predict whether a given digit is "6" (binary classification).
   - Implementation:
     - Data loaded and preprocessed to scale pixel values between 0 and 1.
     - Logistic Regression applied with a maximum iteration of 1000.
     - Evaluated the model using accuracy, classification report, and confusion matrix.
     - Identified misclassified samples for further analysis.
   - PCA for Dimensionality Reduction:
     - Applied PCA to preserve 88% variance of the MNIST dataset.
     - Transformed the training and testing datasets and reevaluated the model.
   - Outcome:
     - Training Accuracy: Achieved high accuracy on the MNIST training dataset.
     - Testing Accuracy: Demonstrated robust generalisation to unseen data.
     - Number of Principal Components Preserved: 88% variance retained.
     - Misclassified Samples: Detailed insights into the misclassified digits.

2. Multiclass Language Prediction:
   - Goal: Predict the language (French, English, Spanish, German) of given text samples.
   - Implementation:
     - TF-IDF Vectorisation: Text samples converted into numeric vectors using TF-IDF.
     - Data split into training and testing sets (80%-20%).
     - Logistic Regression applied for multiclass classification.
     - Evaluated model performance using accuracy and classification metrics.
     - Extracted and analysed misclassified samples and their labels.
   - Outcome:
     - Training Accuracy: Achieved high accuracy on the training dataset.
     - Testing Accuracy: Model showed good generalisation for unseen text samples.
     - Misclassified Labels: Insights into challenging text samples for language prediction.

---

Insights and Analysis:
- PCA reduced the dimensionality of the MNIST dataset effectively while preserving critical information, improving model efficiency.
- Logistic Regression provided high accuracy for both domains and highlighted the need for proper preprocessing and feature extraction.
- Misclassification Analysis:
  - For MNIST, revealed specific digit shapes that confused the model.
  - For language prediction, identified text samples with ambiguous features.

---

Significance:
This project bridges two distinct domains of classification:
- Demonstrates the use of PCA for optimising high-dimensional datasets like MNIST.
- Highlights the role of TF-IDF in processing and classifying text data.
- Provides a robust pipeline for training and evaluating logistic regression models in diverse applications.

---

Skills Highlighted:
- Tools/Technologies: Python, NumPy, TensorFlow, Scikit-learn, PCA, Logistic Regression, TF-IDF Vectoriser.
- Machine Learning Concepts: Dimensionality Reduction, Binary and Multiclass Classification, Text Vectorisation, Model Evaluation Metrics.
- Practical Techniques: Misclassification Analysis, Confusion Matrix Insights, Feature Engineering.
