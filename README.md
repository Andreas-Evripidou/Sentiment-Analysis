# Naive Bayes Sentiment Analyser

This project implements a Naive Bayes sentiment analyser for the Rotten Tomatoes Movie Reviews dataset. The project includes a complete pipeline for preprocessing text data, selecting features, training a Naive Bayes classifier, and evaluating its performance.

## Overview

The Naive Bayes Sentiment Analyser is designed to classify movie reviews into sentiment categories. It includes functionality for:
    Preprocessing text data
    Feature selection
    Training a Naive Bayes classifier
    Evaluating classifier performance with a confusion matrix and macro F1 score

## Getting Started
### Prerequisites

Make sure you have the following Python libraries installed:
    pandas
    numpy
    seaborn
    matplotlib
    nltk

You can install these dependencies using pip:

```bash
pip install requirments.txt
```

### Dataset

The dataset used for training and evaluation is the Rotten Tomatoes Movie Reviews dataset. Ensure your data files are in tab-separated values (TSV) format and include columns for Phrase and Sentiment.
Setup

### Usage
Running the Sentiment Analyser

You can run the sentiment analyser using the following command:

```bash
python NB_sentiment_analyser.py training_data.tsv dev_data.tsv test_data.tsv -classes 5 -features all_words -selection chi_square -nfeatures 1500 -output_files -confusion_matrix
```

Command Line Arguments
    training: Path to the training data file.
    dev: Path to the development data file.
    test: Path to the test data file.
    -classes: Number of sentiment classes (default: 5).
    -features: Feature selection method ("all_words" or "features", default: "all_words").
    -selection: Feature selection method ("most_common" or "chi_square", default: "chi_square").
    -nfeatures: Number of features to be selected (default: 1500).
    -output_files: Whether to save the predictions for dev and test datasets (default: False).
    -confusion_matrix: Whether to print the confusion matrix (default: False).

## Components
### classifier.py

Defines the BayesClassifier class which includes methods for training the Naive Bayes classifier, calculating likelihoods, predicting sentiment classes, and outputting results.

### evaluator.py

Defines the Evaluator class which includes methods for computing the confusion matrix and macro F1 score, and plotting the confusion matrix.

### preprosessor.py

Defines the Preprosessor class which includes methods for preprocessing text data, such as tokenization, stopword removal, and lemmatization. Also includes a method for converting sentiment scores from a five-point scale to a three-point scale.

### feature_selector.py

Defines the Feture_selector class which includes methods for selecting features using either the most common features or chi-square scores.

### NB_sentiment_analyser.py

The main script that integrates all components to perform sentiment analysis. Handles argument parsing, data loading, preprocessing, feature selection, classifier training, prediction, and evaluation.

## Evaluation

The performance of the classifier is evaluated using a confusion matrix and macro F1 score. The confusion matrix provides insights into the classifier's performance across different sentiment classes.
Confusion Matrix

To visualize the confusion matrix, run the script with the -confusion_matrix flag. The confusion matrix will be displayed using a heatmap.
Macro F1 Score

The macro F1 score is calculated based on the confusion matrix and provides a balanced measure of the classifier's performance across all classes.
Results

After running the script, the macro F1 score and (optionally) the confusion matrix will be displayed. If the -output_files flag is set, the predictions for the development and test datasets will be saved to files.
