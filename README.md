(python 3.12.1 was used)
To install the packages needed for the program to work use the following commands:
pip3 install requirments.txt
python3 -m nltk.downloader stopwords punkt wordnet


To run the program enter the following in a Command Prompt or a Terminal:
python3 NB_sentiment_analyser.py <TRAINING_FILE> <DEV_FILE> <TEST_FILE> -classes <NUMBER_CLASSES> -features <all_words,features> -selection <most_common,chi_square> -nfeatures <int> -output_files -confusion_matrix
    
where:

• <TRAINING_FILE> <DEV_FILE> <TEST_FILE> are the paths to the training, dev and test files, respectively.
• -classes <NUMBER_CLASSES> should be either 3 or 5, i.e. the number of classes being predicted.
• -features is a parameter to define whether selected features or no features (i.e. all words) are used. (if this parameter is omitted, all words are used by default)
-selection is a parameter to define the method for feature feature seleciton. (if the paremeter is ommited, chi_square is used by defualt)
-nfeatures is a parameter to define the length of the features list. (if the parameter is omitted, the defualt value is 1500)
• -output_files is an optional value defining whether or not the prediction files should be saved.
• -confusion_matrix is an optional value defining whether confusion matrices should 
