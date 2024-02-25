# Sentiment-Analysis-Using-TF-IDF-and-Logistic-Regression

## Overview
This repository contains code for sentiment analysis using the TF-IDF feature extraction technique and logistic regression classifier. Sentiment analysis aims to determine the sentiment (positive, negative, or neutral) expressed in a piece of text, such as a review, comment, or tweet. In this project, TF-IDF is used to convert text data into numerical features, and logistic regression is employed to classify the sentiment of the text.


## Term Frequency (TF)
Term Frequency (TF) measures the frequency of a term (word) within a document. It indicates how often a particular word occurs in a document relative to the total number of words in that document. TF is calculated using the following formula:

\[ \text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d} \]

Where:
- \( t \) represents a term (word).
- \( d \) represents a document.
- \( \text{TF}(t, d) \) is the Term Frequency of term \( t \) in document \( d \).

TF assigns higher weights to terms that occur more frequently within a document.

## Inverse Document Frequency (IDF)
Inverse Document Frequency (IDF) measures the rarity of a term across the entire corpus of documents. It indicates how important a term is by considering how many documents contain that term. IDF is calculated using the following formula:

\[ \text{IDF}(t) = \log\left(\frac{\text{Total number of documents in the corpus}}{\text{Number of documents containing term } t + 1}\right) \]

Where:
- \( t \) represents a term (word).
- \( \text{IDF}(t) \) is the Inverse Document Frequency of term \( t \).

IDF assigns lower weights to terms that occur in many documents and higher weights to terms that occur in fewer documents. The addition of 1 in the denominator is for smoothing to avoid division by zero in case a term appears in every document.

## TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF combines TF and IDF to calculate the importance of a term in a specific document relative to the entire corpus. It is computed by multiplying the TF of a term in a document by the IDF of that term across all documents. The formula for TF-IDF is as follows:

\[ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) \]

Where:
- \( t ) represents a term (word).
- \( d ) represents a document.
- \( \text{TF-IDF}(t, d) \) is the TF-IDF score of term \( t \) in document \( d \).

TF-IDF assigns higher weights to terms that are frequent in a document but rare in the corpus, indicating their importance in characterizing the content of that document.

## Applications of TF-IDF
- **Information Retrieval**: TF-IDF is used in search engines to rank documents based on their relevance to a given query.
- **Text Mining**: TF-IDF is applied in text mining tasks such as document clustering, classification, and summarization.
- **Keyword Extraction**: TF-IDF can be used to identify important keywords within a document or corpus based on their TF-IDF scores.



## Logistic Regression

### Introduction
Logistic Regression is a supervised learning algorithm used for binary classification tasks. Despite its name, logistic regression is a classification algorithm rather than a regression algorithm. It models the relationship between the input features and the probability of belonging to a particular class.

### Model Representation
In logistic regression, the relationship between the input features \( \mathbf{x} \) and the binary target variable \( y \) is modeled using the logistic function (also known as the sigmoid function). The logistic function maps the linear combination of input features to a probability score between 0 and 1, representing the likelihood of belonging to the positive class (e.g., class 1). The logistic function is defined as follows:

 P(y = 1 | x) = \1/(1 + e^(-w^T.x + b) 

Where:
- \( P(y = 1 |x) is the probability of belonging to the positive class given input features \( \mathbf{x} \).
- \( w ) is the weight vector (coefficients) learned by the logistic regression model.
- \( b ) is the bias term (intercept).
- \( x) is the input feature vector.

### Training
The logistic regression model is trained using labeled training data, where each sample is associated with a binary target variable (e.g., 0 or 1). During training, the model learns the optimal values of the weight vector \( \mathbf{w} \) and the bias term \( b \) by minimizing a loss function, such as the logistic loss or cross-entropy loss. This process involves iteratively adjusting the weights to minimize the difference between the predicted probabilities and the true labels in the training data.

### Prediction
Once the logistic regression model is trained, it can be used to predict the probability of belonging to the positive class for new, unseen samples. Given the input features \( \mathbf{x} \), the model computes the probability of belonging to the positive class using the learned weight vector \( \mathbf{w} \) and bias term \( b \) and the logistic function. The predicted class label is then determined based on whether the predicted probability exceeds a predefined threshold (e.g., 0.5).

### Evaluation
The performance of the logistic regression model can be evaluated using various metrics, such as accuracy, precision, recall, and F1-score. These metrics assess the model's ability to correctly classify samples into their respective classes based on the input features.

### Advantages
- Logistic regression is simple and easy to interpret.
- It performs well when the relationship between input features and target variable is approximately linear.

### Limitations
- Logistic regression assumes a linear relationship between input features and the log-odds of the target variable.
- It may not perform well with highly non-linear data or when there are complex interactions between features.

### Conclusion
Logistic Regression is a versatile and widely used algorithm for binary classification tasks, while TF-IDF is a powerful technique for measuring the importance of terms in a document relative to a collection of documents. Together, they enable effective classification of text data based on its features, making them valuable tools in natural language processing and text analytics applications.


