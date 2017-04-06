# This manual relation extractor finds out the institution relation based on regex patterns.
# It looks for the regular expressions:
# 1) educated at
# 2) graduated from
# 3) matriculated at
# 4) attended
# 5) studied at

import re

if __name__ == "__main__":
    input_file = 'sample_test.tsv'
    output_file = 'output_manual_extractor.tsv'
    output = open(output_file, 'w')

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