# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import pandas as pd


from classifier import BayesClassifier
from evaluator import Evaluator
from preprosessor import Preprosessor



"""
IMPORTANT, modify this part with your details
"""
USER_ID = "acb20ea" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-selection', type=str, default="count_difference", choices=["most_common", "count_difference", "chi_square"])
    parser.add_argument('-nfeatures', type=int, default="1500")
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args

def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    # feature selection method (default = count_difference)
    feature_selection = inputs.selection

    # number of features to be selected (default = 1500)
    nfeatures = inputs.nfeatures

    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    print_confusion_matrix = inputs.confusion_matrix

    train_data = pd.read_csv(training, index_col=0, delimiter='\t') 
    dev_data = pd.read_csv(dev, index_col=0, delimiter='\t') 
    test_data = pd.read_csv(test, index_col=0, delimiter='\t') 

    # Preprosessing phrases
    preprosessor = Preprosessor(number_classes, features, feature_selection)

    if number_classes == 3:
        # Changing scores from 0-4 to 0-2
        train_data["Sentiment"] = train_data["Sentiment"].apply(preprosessor.fiveTothreeScore)
        dev_data["Sentiment"] = dev_data["Sentiment"].apply(preprosessor.fiveTothreeScore)
    
    # Preprosessing phrases 
    train_data['Phrase'] = train_data['Phrase'].apply(preprosessor.preprocess_text) 
    dev_data['Phrase'] = dev_data['Phrase'].apply(preprosessor.preprocess_text) 
    test_data['Phrase'] = test_data['Phrase'].apply(preprosessor.preprocess_text)

    if features == "features":
        # Combine dev and train data to find the most common features
        combined_data = pd.concat([train_data, dev_data])

        # Identify the most crucial features
        preprosessor.feature_selection(combined_data, nfeatures)
        train_data['Phrase'] = train_data['Phrase'].apply(preprosessor.filter_features)
        dev_data['Phrase'] = dev_data['Phrase'].apply(preprosessor.filter_features)
        test_data['Phrase'] = test_data['Phrase'].apply(preprosessor.filter_features)

    # Train the classifier
    classifier = BayesClassifier(train_data, number_classes)
    classifier.train()

    # Predict class for each phrase in dev set
    dev_predictions = {}
    for phrase in dev_data.itertuples():
        dev_predictions[phrase.Index] = classifier.predict_class(phrase.Phrase)

    # Create confusion matrix
    evaluator = Evaluator(dev_data, dev_predictions, number_classes, features)
    confusion_matrix = evaluator.compute_confusion_matrix()

    # Calculate macro F1 score
    f1_score = evaluator.compute_macro_f1_score(confusion_matrix)
    
    # If confusion_matrix is set to True, plot the confusion matrix
    if print_confusion_matrix:
        evaluator.plot_confiusion_matrix(confusion_matrix)
    
    # If output_files is set to True, run the classification on the test data and 
    # save the results of the dev and test data to files
    if output_files:
        classifier.output("dev_predictions_" + str(number_classes) + "classes_" + USER_ID + ".tsv", dev_predictions)
        
        # Predict class for each phrase in test set
        test_predictions = {}
        for phrase in test_data.itertuples():
            test_predictions[phrase.Index] = classifier.predict_class(phrase.Phrase)

        classifier.output("test_predictions_" + str(number_classes) + "classes_" + USER_ID + ".tsv", test_predictions)  
        
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()