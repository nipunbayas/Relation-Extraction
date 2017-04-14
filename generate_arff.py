from nltk.tokenize import sent_tokenize
from nltk.parse.stanford import StanfordDependencyParser

import re
import os

os.environ['STANFORD_PARSER'] = '/home/nipun/nltk_data'
os.environ['STANFORD_MODELS'] = '/home/nipun/nltk_data'

TEST_DATA_PATH = "test.tsv"
TRAIN_DATA_PATH = "train.tsv"
BROWN_CLUSTER_FILE = "brown_cluster_paths"
# Use extractor type as either - 'manual', 'normal', 'nltk_tokenizer', 'brown', 'dependency_features' or 'kitchen_sink'
extractor = 'dependency_features'
cluster_prefix_length = 5


def manual_relation_extractor():
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    with open(TEST_DATA_PATH) as inputted_file:
        lines = inputted_file.read().splitlines()  # read input file

    for line in lines:
        sentences = line.split('\t')
        sentence = sentences[2]
        gold_standard_result = sentences[-1]

        match = re.search(r'\beducated at\b', sentence, flags=re.IGNORECASE) or \
                re.search(r'\bgraduated from\b', sentence, flags=re.IGNORECASE) or \
                re.search(r'\bmatriculated at\b', sentence, flags=re.IGNORECASE) or \
                re.search(r'\battended\b', sentence, flags=re.IGNORECASE) or \
                re.search(r'\bstudied at\b', sentence, flags=re.IGNORECASE)

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


def get_brown_clusters():
    brown_clusters = {}
    with open(BROWN_CLUSTER_FILE) as inputted_file:
        data_lines = inputted_file.read().splitlines()

    for line in data_lines:
        word_cluster = line.split('\t')
        brown_clusters[word_cluster[1]] = word_cluster[0]

    return brown_clusters


def parse_data(train_data, test_data, extractor):
    """
    Input: path to the data file
    Output: (1) a list of tuples, one for each instance of the data, and
            (2) a list of all unique tokens in the data

    Parses the data file to extract all instances of the data as tuples of the form:
    (person, institution, judgment, full snippet, intermediate text)
    where the intermediate text is all tokens that occur between the first occurrence of
    the person and the first occurrence of the institution.

    Also extracts a list of all tokens that appear in the intermediate text for the
    purpose of creating feature vectors.
    """
    all_tokens = []
    data = []
    for fp in [train_data, test_data]:
        with open(fp) as f:
            for line in f:
                institution, person, snippet, intermediate_text, judgment = line.split("\t")
                judgment = judgment.strip()

                # Build up a list of unique tokens that occur in the intermediate text
                # This is needed to create BOW feature vectors
                if extractor is 'normal':
                    tokens = intermediate_text.split()
                # Using the NTLK Sentence Tokenizer
                elif extractor is 'nltk_tokenizer':
                    intermediate_text = unicode(intermediate_text, errors='replace')
                    tokens = sent_tokenize(intermediate_text)
                # Using Brown Clusters
                elif extractor is 'brown':
                    tokens = []
                    brown_cluster_dict = get_brown_clusters()
                    words = intermediate_text.split()
                    for word in words:
                        if word in brown_cluster_dict:
                            tokens.append(brown_cluster_dict[word][0:cluster_prefix_length])
                # Using Stanford Dependency Parser
                elif extractor is 'dependency_features':
                    intermediate_text = unicode(intermediate_text, errors='replace')
                    dep_parser = StanfordDependencyParser(model_path='/home/nipun/nltk_data/englishPCFG.ser.gz', java_options = '-mx4096m')
                    tokens = dep_parser.raw_parse(intermediate_text)
                    for t in tokens:
                        print t.tree()
                        print " "
                """ To generate CoNLL file, type the commands:
                      java -mx150m -cp "stanford-parser-full-2016-10-31/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "penn" /home/nipun/nltk_data/englishPCFG.ser.gz intermediate_text >testsent.tree

java -mx150m -cp "stanford-parser-full-2016-10-31/*:" edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile testsent.tree -conllx """

                for t in tokens:
                    if extractor is not 'brown':
                        t = t.lower()
                    if t not in all_tokens:
                        all_tokens.append(t)
                data.append((person, institution, judgment, snippet, intermediate_text))
    return data, all_tokens


def create_feature_vectors(data, all_tokens, extractor):
    """
    Input: (1) The parsed data from parse_data()
             (2) a list of all unique tokens found in the intermediate text
    Output: A list of lists representing the feature vectors for each data instance

    Creates feature vectors from the parsed data file. These features include
    bag of words features representing the number of occurrences of each
    token in the intermediate text (text that comes between the first occurrence
    of the person and the first occurrence of the institution).
    This is also where any additional user-defined features can be added.
    """
    feature_vectors = []
    for instance in data:
        # BOW features
        # Gets the number of occurrences of each token
        # in the intermediate text
        feature_vector = [0 for t in all_tokens]
        intermediate_text = instance[4]

        if extractor is 'normal':
            tokens = intermediate_text.split()
        # Using NLTK Sentence Tokenizer
        elif extractor is 'nltk_tokenizer':
            tokens = sent_tokenize(intermediate_text)
        # Using Brown clusters
        elif extractor is 'brown':
            tokens = []
            brown_cluster_dict = get_brown_clusters()
            words = intermediate_text.split()
            for word in words:
                if word in brown_cluster_dict:
                    tokens.append(brown_cluster_dict[word][0:cluster_prefix_length])
        # Using Stanford Dependency Parser
        elif extractor is 'dependency_features':
            dep_parser = StanfordDependencyParser(model_path='/home/nipun/nltk_data/englishPCFG.ser.gz', java_options = '-mx20000m')
            tokens = dep_parser.raw_parse(intermediate_text)
            
        
        for token in tokens:
            if extractor is 'brown':
                index = all_tokens.index(token)
            else:
                index = all_tokens.index(token.lower())
            feature_vector[index] += 1

        ### ADD ADDITIONAL FEATURES HERE ###

        # Class label
        judgment = instance[2]
        feature_vector.append(judgment)

        feature_vectors.append(feature_vector)
    return feature_vectors


def generate_arff_file(feature_vectors, all_tokens, out_path, extractor):
    """
    Input: (1) A list of all feature vectors for the data
             (2) A list of all unique tokens that occurred in the intermediate text
             (3) The name and path of the ARFF file to be output
    Output: an ARFF file output to the location specified in out_path

    Converts a list of feature vectors to an ARFF file for use with Weka.
    """
    with open(out_path, 'w') as f:
        # Header info
        f.write("@RELATION institutions\n")
        for i in range(len(all_tokens)):
            if extractor is 'brown':
                f.write("@ATTRIBUTE cluster_{} integer\n".format(i))
            else:
                f.write("@ATTRIBUTE token_{} integer\n".format(i))

        ### SPECIFY ADDITIONAL FEATURES HERE ###
        # For example: f.write("@ATTRIBUTE custom_1 REAL\n")

        # Classes
        f.write("@ATTRIBUTE class {yes,no}\n")

        # Data instances
        f.write("\n@DATA\n")
        for fv in feature_vectors:
            features = []
            for i in range(len(fv)):
                value = fv[i]
                if value != 0:
                    features.append("{} {}".format(i, value))
            entry = ",".join(features)
            f.write("{" + entry + "}\n")

if __name__ == "__main__":
    data, all_tokens = parse_data(TRAIN_DATA_PATH, TEST_DATA_PATH, extractor)
    feature_vectors = create_feature_vectors(data, all_tokens, extractor)
    generate_arff_file(feature_vectors[:6000], all_tokens, "train.arff", extractor)
    generate_arff_file(feature_vectors[6000:], all_tokens, "test.arff", extractor)
