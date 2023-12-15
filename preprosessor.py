
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
     

class Preprosessor:

    def __init__(self, number_of_classes, features=None):
        """
        Constructor for the Preprocessor class.

        Parameters:
        - features: List of features to be used in preprocessing (optional).
        """
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

        # If not using all features, remove stopwords and lem the words and remove special characters and numbers
        if self.features == "features":
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            tokens = [lemmatizer.lemmatize(token) for token in tokens if (token.isalpha() and token not in stop_words)]

        else:
            # Else removing special characters and numbers
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

    