# This manual relation extractor finds out the institution relation based on regex patterns.
# It looks for the regular expressions:
# 1) educated at
# 2) graduated from
# 3) matriculated at
# 4) attended
# 5) studied at

import re
import pandas as pd

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

if __name__ == "__main__":
    input_file = 'test.tsv'
    output_file = 'output_manual_extractor.tsv'
    output = open(output_file, 'w')

    test_model_file = load_files(input_file, encoding='latin-1')

    with open(input_file) as inputted_file:
        lines = inputted_file.read().splitlines()  # read input file

    for line in lines:
        sentences = line.split('\t')
        sentence = sentences[2]

        match = re.search(r'\beducated at\b', sentence, flags=re.IGNORECASE) or \
            re.search(r'\bgraduated from\b', sentence, flags=re.IGNORECASE) or \
            re.search(r'\bmatriculated at\b', sentence, flags=re.IGNORECASE) or \
            re.search(r'\battended\b', sentence, flags=re.IGNORECASE) or \
            re.search(r'\bstudied at\b', sentence, flags=re.IGNORECASE)
        if match:
            output.write("%s\t%s\t%s\t%s\t%s\n" % (sentences[0], sentences[1], sentences[2], sentences[3], 'yes'))
        else:
            output.write("%s\t%s\t%s\t%s\t%s\n" % (sentences[0], sentences[1], sentences[2], sentences[3], 'no'))
    output.close()

    test_generated_file = load_files(output_file, encoding='latin-1')

    vectors = CountVectorizer.fit_transform(test_model_file.data)
    print vectors.shape
