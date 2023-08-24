# Spam Detector

A machine learning model to detect spam messages using Natural Language Processing (NLP). The project utilizes Multinomial Naive Bayes classifier and various metrics to evaluate the model's performance.

## Overview

The Spam Detector reads a dataset containing labeled spam and ham (non-spam) messages. It preprocesses the data, splits it into training and testing sets, and trains a Multinomial Naive Bayes model to classify messages as spam or ham.

## Features

- **Data Preprocessing**: Cleans and encodes the data, converting text messages into numerical format.
- **Model Training**: Trains a Multinomial Naive Bayes classifier on the training data.
- **Evaluation Metrics**: Evaluates the model using accuracy, F1 score, AUC, and confusion matrix.
- **Visualization**: Includes word cloud visualizations for spam and ham messages.
- **Misclassification Analysis**: Identifies messages that were misclassified by the model.

## Dependencies

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- wordcloud

## Usage

1. Load the dataset containing spam and ham messages.
2. Run the code to preprocess the data, train the model, and evaluate its performance.
3. Visualize the common words in spam and ham messages.
4. Analyze any misclassified messages.

## Code Snippet

```python
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train acc:", model.score(Xtrain, Ytrain))
print("test acc:", model.score(Xtest, Ytest))
