import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, correct_labels, predicted_labels, number_classes, features):
        """
        Constructor for the Evaluator class.

        Parameters:
        - correct_labels: Dictionary containing correct sentiment labels for each phrase_index.
        - predicted_labels: Dictionary containing predicted sentiment labels for each phrase_index.
        - number_classes: Number of classes (sentiment categories).
        - features: Features used in the evaluation.
        """
        self.correct_labels = correct_labels
        self.predicted_labels = predicted_labels
        self.number_classes = number_classes
        self.features = features
    
    def compute_confusion_matrix(self):
        """
        Computes the confusion matrix based on correct and predicted labels.

        Returns:
        - confusion_matrix: NumPy array representing the confusion matrix.
        """
        confusion_matrix = np.zeros((self.number_classes, self.number_classes))
        for phrase_index, predicted_class in self.predicted_labels.items():
            correct = self.correct_labels['Sentiment'][phrase_index]
            confusion_matrix[correct][predicted_class] += 1
        return confusion_matrix
    
    def compute_macro_f1_score(self, confusion_matrix):
        """
        Computes the macro F1 score based on the confusion matrix.

        Parameters:
        - confusion_matrix: NumPy array representing the confusion matrix.

        Returns:
        - macro_f1: Macro F1 score.
        """
        macro_f1 = 0
        for i in range(self.number_classes):
            true_positive = confusion_matrix[i][i]
            fasle_postive = confusion_matrix[:, i].sum() - true_positive
            false_negative = confusion_matrix[i, :].sum() - true_positive
            f1 = 2 * true_positive / (2 * true_positive + fasle_postive + false_negative)
            macro_f1 += f1

        macro_f1 /= self.number_classes
        return macro_f1
    
    
    def plot_confiusion_matrix(self, confusion_matrix):
        """
        Plots the confusion matrix using seaborn.

        Parameters:
        - confusion_matrix: NumPy array representing the confusion matrix.
        """

        # Define class labels based on the number of classes.
        if self.number_classes == 5:
            labels = ['Negative', 'Somewhat Negative', 'Neutral', 'Somewhat Positive', 'Positive']
        # Plot the confusion matrix using seaborn heatmap
        elif self.number_classes == 3:
            labels = ['Negative', 'Neutral', 'Positive']
        axes = sns.heatmap(confusion_matrix, annot=True, fmt='.3g', xticklabels=labels, yticklabels=labels)
        axes.set(xlabel='Predicted', ylabel='Actual')
        axes.set_title('Confusion Matrix: ' + str(self.number_classes) + ' classes, ' + str(self.features))
        plt.tight_layout()
        plt.show()

