import pandas as pd

class BayesClassifier:

     # Creates new Classify object storing the number of classes
    def __init__(self, train_data: pd.DataFrame, number_classes):
        """
        Constructor for the BayesClassifier class.

        Parameters:
        - train_data: DataFrame containing training data with 'Phrase' and 'Sentiment' columns.
        - number_classes: Number of sentiment classes.
        """
        self.train_data = train_data
        self.number_classes = number_classes

    def train(self):
        """
        Trains the Naive Bayes classifier using the provided training data.
        """
        # Create dictionary for each class
        class_dicts = []
        for i in range(self.number_classes):
            class_dicts.append({"total": 0})
        
        # Add words to each class dictionary
        for phrase in self.train_data.itertuples():
            class_dicts[phrase.Sentiment]["total"] += 1
            for word in phrase.Phrase.split():
                if word in class_dicts[phrase.Sentiment]:
                    class_dicts[phrase.Sentiment][word] += 1
                else:
                    class_dicts[phrase.Sentiment][word] = 1

        # Save clas_dict for later use
        self.class_dicts = class_dicts

        # Calculate total number of phrases
        total = len(self.train_data)
        
        # Combine all dictionaries
        combined_dict = {}
        for class_dict in class_dicts:
            combined_dict.update(class_dict)
        # Calculate number of unique words
        V = len(combined_dict) - self.number_classes

        # Calculate prior for each class
        self.priors = []
        for class_dict in class_dicts:
            self.priors.append(self.calculate_prior(class_dict, total))

        # Calculate likelihood for each word in each class
        self.likelihoods = []
        for class_dict in class_dicts:
            likelihood = {}
            for word in class_dict:
                likelihood[word] = self.calcuate_likelihood(class_dict, word, V)
            self.likelihoods.append(likelihood)
    

    # Function to calculate the likelihood of a word given a class
    def calcuate_likelihood(self, class_dict, word , V):
        """
        Calculates the likelihood of a word given a class.

        Parameters:
        - class_dict: Dictionary containing word frequencies for a class.
        - word: Word for which likelihood is calculated.
        - V: Number of unique words.

        Returns:
        - likelihood: Calculated likelihood.
        """
        if word in class_dict:
            return (class_dict[word] + 1)/(class_dict["total"] + V)
        else:
            return 1/(self.class_dicts["total"] + V)

    def get_likelihood(self, word, sentiment_class):
        """
        Gets the likelihood of a word given a sentiment class.

        Parameters:
        - word: Word for which likelihood is retrieved.
        - sentiment_class: Sentiment class index.

        Returns:
        - likelihood: Likelihood of the word in the given class.
        """
        if word in self.likelihoods[sentiment_class]:
            return self.likelihoods[sentiment_class][word]
        else:
            return 1/(len(self.likelihoods[sentiment_class]) + 1)
        
    def calculate_prior(self, class_dict, total):
        """
        Calculates the prior probability for a class.

        Parameters:
        - class_dict: Dictionary containing word frequencies for a class.
        - total: Total number of phrases.

        Returns:
        - prior: Calculated prior probability for the class.
        """
        return class_dict["total"]/total
    
    def predict_class(self, phrase):
        """
        Predicts the sentiment class for a given phrase using Naive Bayes.

        Parameters:
        - phrase: Input phrase to be classified.

        Returns:
        - predicted: Predicted sentiment class.
        """
        max_prob = 0
        predicted = None
        for sentiment_class in range(self.number_classes):
            prob = self.priors[sentiment_class]
            for word in phrase.split():
                prob *= self.get_likelihood(word, sentiment_class)
            if prob > max_prob:
                max_prob = prob
                predicted = sentiment_class
        return predicted
    
    def output(self, outfile, predicted_classes):
        """
        Outputs the predicted classes to a file.

        Parameters:
        - outfile: Path to output file.
        - predicted_classes: Dictionary containing predicted sentiment classes for each phrase_index.
        """
        with open(outfile, 'w') as out:
            print("%s\t%s" % ("SentenceID", "Sentiment"), file=out)
            for sentenceID, sentiment in predicted_classes.items():
                print("%d\t%d" % (sentenceID, sentiment), file=out)