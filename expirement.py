import subprocess

def run_command(training, dev, test, classes, features, selection, output_files, confusion_matrix, nfeatures):
    command = [
        "python3.12.exe",
        "NB_sentiment_analyser.py",
        training,
        dev,
        test,
        "-classes", str(classes),
        "-features", features,
        "-selection", selection,
        "-nfeatures", str(nfeatures)
    ]

    if output_files:
        command.append("-output_files")

    if confusion_matrix:
        command.append("-confusion_matrix")
    # run command and return command output
    result = subprocess.check_output(command, text=True)
    
    return result

    

# Example usage:
if __name__ == "__main__":
    argument_sets_with_all_words = [
        {"training": "./moviereviews/train.tsv", "dev": "./moviereviews/dev.tsv", "test": "./moviereviews/test.tsv", "classes": 3, "features": "all_words", "selection": "count_difference", "output_files": False, "confusion_matrix": False},
        {"training": "./moviereviews/train.tsv", "dev": "./moviereviews/dev.tsv", "test": "./moviereviews/test.tsv", "classes": 5, "features": "all_words", "selection": "count_difference", "output_files": False, "confusion_matrix": False},
    ]
    # Define multiple sets of arguments
    argument_sets = [
        {"training": "./moviereviews/train.tsv", "dev": "./moviereviews/dev.tsv", "test": "./moviereviews/test.tsv", "classes": 3, "features": "features", "selection": "most_common", "output_files": False, "confusion_matrix": False},
        {"training": "./moviereviews/train.tsv", "dev": "./moviereviews/dev.tsv", "test": "./moviereviews/test.tsv", "classes": 3, "features": "features", "selection": "count_difference", "output_files": False, "confusion_matrix": False},
        {"training": "./moviereviews/train.tsv", "dev": "./moviereviews/dev.tsv", "test": "./moviereviews/test.tsv", "classes": 3, "features": "features", "selection": "chi_square", "output_files": False, "confusion_matrix": False},
        {"training": "./moviereviews/train.tsv", "dev": "./moviereviews/dev.tsv", "test": "./moviereviews/test.tsv", "classes": 5, "features": "features", "selection": "most_common", "output_files": False, "confusion_matrix": False},
        {"training": "./moviereviews/train.tsv", "dev": "./moviereviews/dev.tsv", "test": "./moviereviews/test.tsv", "classes": 5, "features": "features", "selection": "count_difference", "output_files": False, "confusion_matrix": False},
        {"training": "./moviereviews/train.tsv", "dev": "./moviereviews/dev.tsv", "test": "./moviereviews/test.tsv", "classes": 5, "features": "features", "selection": "chi_square", "output_files": False, "confusion_matrix": False},

    ]
    # Results talbe columns: classes, features, selection, nfeatures. columns: macrof1
    

    results = []
    # Run all_word argument sets
    for arguments in argument_sets_with_all_words:
        result = run_command(**arguments, nfeatures=1000)
        results.append("{classes:^8}|{features:^10}|{selection:^16}|{nfeatures:^11}|{macrof1:^8}".format(**arguments, nfeatures = 0, macrof1=result.split()[-1]))

    for n in [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1500, 2000]:
        # Run the script with different argument sets
        for arguments in argument_sets:
            result = run_command(**arguments,nfeatures=n)
            
            results.append("{classes:^8}|{features:^10}|{selection:^16}|{nfeatures:^11}|{macrof1:^8}".format(**arguments,nfeatures = n, macrof1=result.split()[-1]))

    # Print results table
    print("classes | features |   selection    | nfeatures | macrof1")
    print("--------+----------+----------------+-----------+--------")
    for result in results:
        print(result)