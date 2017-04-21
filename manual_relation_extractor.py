# This manual relation extractor finds out the institution relation based on regex patterns.
# It looks for the regular expressions:
# 1) educated
# 2) graduated
# 3) taught
# 4) attended
# 5) studied

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
        pos_sentence = ''.join(str(e) for e in pos_tags_list)
        pos_sentence = pos_sentence.replace('(', ' ')
        pos_sentence = pos_sentence.replace(')', ' ')
        pos_sentence = pos_sentence.replace('u\'', ' ')
        pos_sentence = pos_sentence.replace('\',', ' ')
        pos_sentence = pos_sentence.replace('  ', ' ')
       
        gold_standard_result = sentences[-1]

        match = re.search(r'\bgraduated\b', sentence, flags=re.IGNORECASE) or \
                re.search(r'\battended\b', sentence, flags=re.IGNORECASE) or \
                re.search(r"'PRP'(\s)+([a-zA-Z0-9\.]+)(\s)+'VBD'(\s)+([a-zA-Z0-9\.]+)(\s)+'IN'", pos_sentence, flags=re.IGNORECASE) or \
                re.search(r"'NNP'(\s)+([a-zA-Z0-9\.]+)(\s)+'VBD'(\s)+([a-zA-Z0-9\.]+)(\s)+'NNP'", pos_sentence, flags=re.IGNORECASE) or \
                re.search(r"'NNP'(\s)+([a-zA-Z0-9\.]+)(\s)+'studied'(\s)+([a-zA-Z0-9\.]+)(\s)+'IN'", pos_sentence, flags=re.IGNORECASE)
        #    re.search(r'\bgraduated\b', sentence, flags=re.IGNORECASE) or \
        #    re.search(r'\btaught\b', sentence, flags=re.IGNORECASE) or \
        #    re.search(r'\battended\b', sentence, flags=re.IGNORECASE) or \
        #    re.search(r'\bstudied\b', sentence, flags=re.IGNORECASE) or \

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
    print "Precision: ", precision
    print "Recall: ", recall
    print "F1 Score: ", f1_score
