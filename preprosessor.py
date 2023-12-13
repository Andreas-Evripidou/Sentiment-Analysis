
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
     

class Preprosessor:

    def __init__(self, number_of_classes, features=None, feature_selection="most_common"):
        """
        Constructor for the Preprocessor class.

        Parameters:
        - features: List of features to be used in preprocessing (optional).
        """
        self.feature_selection_methods = {
            "most_common": self.most_common_features,
            "count_difference": self.count_difference_features,
            "chi_square": self.chi_square_features
        }
        self.feature_method = feature_selection
        self.features = features
        self.number_of_classes = number_of_classes

    def preprocess_text(self, text):
        """
        Preprocesses the input text by lowercasing, tokenizing, removing special characters and numbers, and lemmatizing.

        Parameters:
        - text: Input text to be preprocessed.

        Returns:
        - preprocessed_text: Preprocessed text.
        """
        # Lowercasing
        text = (text.lower().replace("n't", "not")
                                .replace("'s", "is")
                                .replace("'m", "am")
                                .replace("'re", "are")
                                .replace("'ll", "will")
                                .replace("'ve", "have")
                                .replace("'d", "would"))

        # Tokenization
        tokens = word_tokenize(text)

        # If not using all features, remove stopwords and stem the words
        if self.features == "features":
            # Remove stopwords stem the words
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.lower() not in stop_words]
        

        # Removing special characters and numbers
        tokens = [token for token in tokens if token.isalpha()]

        # Joining tokens back into text
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

    def fiveTothreeScore(slef, score):
        """
        Converts a five-point sentiment score to a three-point scale.

        Parameters:
        - score: Input sentiment score (0 to 4).

        Returns:
        - converted_score: Transformed sentiment score (0, 1 or 2).
        """
        if score < 2:
            return 0
        elif score >= 3:
            return 1
        else:
            return 2

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
    
    def count_difference_features(self, data, number_of_features):
        """
            Selects the features with the largest difference in frequency between classes.

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
        
        # Calculate difference in frequency between classes
        differences = []
        for word in word_freq:
            freq = []
            for i in range(self.number_of_classes):
                if word in self.class_dicts[i]:
                    freq.append(self.class_dicts[i][word])
            differences.append((word, max(freq) - min(freq)))
        
        # Sort features by difference in frequency
        differences.sort(key=lambda x: x[1], reverse=True)
        most_common = [word for word, freq in differences[:number_of_features]]
        
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
        

    