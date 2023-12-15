from collections import Counter
from nltk.tokenize import word_tokenize

class Feture_selector:
    def __init__(self, feature_method, number_of_classes=5):
        self.feature_selection_methods = {
            "most_common": self.most_common_features,
            "chi_square": self.chi_square_features
        }
        self.feature_method = feature_method
        self.number_of_classes = number_of_classes

    def feature_selection(self, data, number_of_features):
        """
            Selects the most common features from the training data.
        """
        class_dicts = []
        for i in range(self.number_of_classes):
            class_dicts.append({"total": 0})

        # Add words to each class dictionary
        for phrase in data.itertuples():
            class_dicts[phrase.Sentiment]["total"] += 1
            for word in phrase.Phrase.split():
                if word in class_dicts[phrase.Sentiment]:
                    class_dicts[phrase.Sentiment][word] += 1
                else:
                    class_dicts[phrase.Sentiment][word] = 1

        # Save clas_dict for later use
        self.class_dicts = class_dicts

        self.selected_features =  self.feature_selection_methods[self.feature_method](data, number_of_features)
    
    def most_common_features(self, data, number_of_features):
        """
            Selects the most common features from the training data.

            Parameters:
            - data: Data to be used for feature selection.
            - number_of_features: Number of features to be selected.

            Returns:
            - most_common: List of most common features.
        """
        word_freq = Counter()
        # Add words to each class dictionary
        for phrase in data.itertuples():
            word_freq.update(phrase.Phrase.split())
        most_common = [word for word, freq in word_freq.most_common(number_of_features)]
        
        return most_common
    
    def chi_square_features(self, data, number_of_features):
        """
            Selects the features with the highest chi-square score.

            Parameters:
            - data: Data to be used for feature selection.
            - number_of_features: Number of features to be selected.

            Returns:
            - most_common: List of most common features.
        """
        word_freq = Counter()
        # Add words to each class dictionary
        for phrase in data.itertuples():
            word_freq.update(phrase.Phrase.split())
        
        # Calculate chi-square score for each feature
        chi_square = []
        for word in word_freq:
            freq = []
            for i in range(self.number_of_classes):
                if word in self.class_dicts[i]:
                    freq.append(self.class_dicts[i][word])
                else:
                    freq.append(0)
            chi_square.append((word, self.chi_square(freq)))
        
        # Sort features by chi-square score
        chi_square.sort(key=lambda x: x[1], reverse=True)
        most_common = [word for word, freq in chi_square[:number_of_features]]
        
        return most_common

    def chi_square(self, freq):
        """
            Calculates the chi-square score for a feature.

            Parameters:
            - freq: List of frequencies for a feature.

            Returns:
            - chi_square: Chi-square score for the feature.
        """
        chi_square = 0
        for i in range(self.number_of_classes):
            expected = sum(freq) * self.class_dicts[i]["total"] / sum(self.class_dicts[i].values())
            chi_square += (freq[i] - expected)**2 / expected

        return chi_square

    
    def filter_features(self, text):
        # Filter the data based on the selected features
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token in self.selected_features]

        # Joining tokens back into text
        filtered_phrase = ' '.join(filtered_tokens)

        return filtered_phrase