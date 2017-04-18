# This manual relation extractor finds out the institution relation based on regex patterns.
# It looks for the regular expressions:
# 1) educated at
# 2) graduated from
# 3) matriculated at
# 4) attended
# 5) studied at

from __future__ import division
from nltk import word_tokenize, pos_tag
import re

INPUT_FILE = 'test.tsv'

if __name__ == "__main__":
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    pos_tags_list = list()

    with open(INPUT_FILE) as inputted_file:
        lines = inputted_file.read().splitlines()  # read input file

    for line in lines:
        sentences = line.split('\t')
        sentence = sentences[2]

        unicode_sentence = unicode(sentence, errors='replace')
        text = word_tokenize(unicode_sentence)
        pos_tags_list = pos_tag(text)
        print pos_tags_list
        gold_standard_result = sentences[-1]

        match = re.search(r'\beducated\b', sentence, flags=re.IGNORECASE) or \
            re.search(r'\bgraduated\b', sentence, flags=re.IGNORECASE) or \
            re.search(r'\bmatriculated\b', sentence, flags=re.IGNORECASE) or \
            re.search(r'\battended\b', sentence, flags=re.IGNORECASE) or \
            re.search(r'\bstudied\b', sentence, flags=re.IGNORECASE)

        if match:
            if gold_standard_result == 'yes':
                true_positives += 1
            else:
                false_positives += 1
        else:
            if gold_standard_result == 'no':
                true_negatives += 1
            else:
                false_negatives += 1

    print true_positives, ", ", true_negatives, ", ", false_positives, ", ", false_negatives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print precision, recall, f1_score