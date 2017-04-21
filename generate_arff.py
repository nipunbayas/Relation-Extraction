from nltk.tokenize import sent_tokenize, wordpunct_tokenize, word_tokenize, sent_tokenize
from nltk.parse.stanford import StanfordDependencyParser

import os
import nltk

os.environ['STANFORD_PARSER'] = '/home/nipun/nltk_data'
os.environ['STANFORD_MODELS'] = '/home/nipun/nltk_data'

TEST_DATA_PATH = 'test.tsv'
TRAIN_DATA_PATH = 'train.tsv'
BROWN_CLUSTER_FILE = 'brown_cluster_paths'
CONLL_FILE = 'sentences.conll'

# Use extractor type as either - 'bow', 'nltk_tokenizer', 'brown', 'brown_full', 'dependency_features' or 'kitchen_sink'
extractor = 'kitchen_sink'

if extractor == 'brown_full':
    cluster_prefix_length = 8
elif extractor == 'brown' or extractor is 'kitchen_sink':
    cluster_prefix_length = 6

def manual_regex_feature():
    regex_features = list()
    with open(TRAIN_DATA_PATH) as inputted_file:
        lines = inputted_file.read().splitlines()  # read input file

    for line in lines:
        sentences = line.split('\t')
        if sentences[-1] == 'yes':
            regex_features.append(1)
        else:
            regex_features.append(0)

    with open(TEST_DATA_PATH) as inputted_file:
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

        if match:
            regex_features.append(1)
        else:
            regex_features.append(0)

    return regex_features


def generate_dependency_features():
    institution_relation_words = ['graduated', 'studied', 'matriculated', 'attended', 'educated',
                                          'completed', 'taught', 'received']
    feature1_list = list()
    # Feature 1: Check if root word is one of the 'institution_relation_words'
    # If yes, store yes, else store 'no'
    words = list()
    # Feature 2: Find conj_and, prep_as and x_comp in the labels. They generally represent institution relation
    pos_tag_list = list()
    feature2_list = list()
    feature2_int_list = list()
    # Feature 3: For the root node, count the number of labels for all its children
    label_list = list()
    root_list = list()
    feature3_list = list()
    label_children = 0
    feature1 = ""

    with open(CONLL_FILE) as conll_input:
        for line in conll_input:
            parse_tree = line.split('\t')
            if len(parse_tree) != 1:
                pos_tag = parse_tree[7]
                if parse_tree[7] == 'root':
                    root_word = parse_tree[1]
                    root_word_index = parse_tree[0]

                words.append(parse_tree[1])
                pos_tag_list.append(parse_tree[3])
                label_list.append(parse_tree[7])
                root_list.append(parse_tree[6])                     
            else:
                words = list()

                if root_word in institution_relation_words:
                    feature1 = 1
                else:
                    feature1 = 0
                feature1_list.append(feature1)

                for label in label_list:
                    if label == 'conj_and' or label == 'prep_as' or label == 'x_comp':
                        feature2 = 1
                        break
                    else:
                        feature2 = 0

                feature2_list.append(feature2)
                
                count = 0
                for index in root_list:
                    if index == root_word_index:
                        label_children += 1
                feature3_list.append(label_children)
                verb_count = 0
                label_children = 0

    return feature1_list, feature2_list, feature3_list


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
                if extractor is 'bow' or extractor is 'kitchen_sink':
                    tokens = intermediate_text.split()
                # Using the NTLK Sentence Tokenizer
                elif extractor is 'nltk_tokenizer' or extractor is 'kitchen_sink':
                    intermediate_text = unicode(intermediate_text, errors='replace')
                    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
                    if extractor is 'kitchen_sink':
                        tokens.append(tokenizer.tokenize(intermediate_text))
                    else:
                        tokens = tokenizer.tokenize(intermediate_text)
                    # tokens = sent_tokenize(intermediate_text)
                # Using Brown Clusters
                elif extractor is 'brown' or extractor is 'brown_full' or extractor is 'kitchen_sink':
                    if extractor is not 'kitchen_sink':
                        tokens = []
                    brown_cluster_dict = get_brown_clusters()
                    words = intermediate_text.split()
                    for word in words:
                        if word in brown_cluster_dict:
                            tokens.append(brown_cluster_dict[word][0:cluster_prefix_length])

                """ To generate CoNLL file, type the commands:
                      java -mx2048m -cp "../stanford-parser-full-2016-10-31/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences "newline" -maxLength "250" -outputFormat "penn" /home/nipun/nltk_data/englishPCFG.ser.gz sentence.txt >testsent.tree

java -mx12048m -cp "../stanford-parser-full-2016-10-31/*:" edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile testsent.tree -conllx > sentence.conll"""
                if extractor is not 'dependency_features':
                    for t in tokens:
                        if extractor is 'bow' or extractor is 'nltk_tokenizer':
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
    judgement_list = []
    for instance in data:
        # BOW features
        # Gets the number of occurrences of each token
        # in the intermediate text
        feature_vector = [0 for t in all_tokens]
        
        intermediate_text = instance[4]
        judgement_list.append(instance[2])

        if extractor is 'bow' or extractor is 'kitchen_sink':
            tokens = intermediate_text.split()
        # Using NLTK Sentence Tokenizer
        elif extractor is 'nltk_tokenizer' or extractor is 'kitchen_sink':
            tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
            if extractor is 'nltk_tokenizer':
                tokens = tokenizer.tokenize(intermediate_text)
                # tokens = sent_tokenize(intermediate_text)
            else:
                tokens.append(tokenizer.tokenize(intermediate_text))
        # Using Brown clusters
        elif extractor is 'brown' or extractor is 'brown_full' or extractor is 'kitchen_sink':
            if extractor is not 'kitchen_sink':
                tokens = []
            brown_cluster_dict = get_brown_clusters()
            words = intermediate_text.split()
            for word in words:
                if word in brown_cluster_dict:
                    tokens.append(brown_cluster_dict[word][0:cluster_prefix_length])
        
        if extractor is not 'dependency_features':
            for token in tokens:
                if extractor is 'brown' or extractor is 'brown_full':
                    index = all_tokens.index(token)
                else:
                    try:
                        index = all_tokens.index(token.lower())
                    except:
                        pass
                feature_vector[index] += 1

            ### ADD ADDITIONAL FEATURES HERE ###

            # Class label
            if extractor is not 'kitchen_sink':
                judgment = instance[2]
                feature_vector.append(judgment)

            feature_vectors.append(feature_vector)

    if extractor is 'dependency_features' or extractor is 'kitchen_sink':
        feature1, feature2, feature3 = generate_dependency_features()
        if extractor is 'dependency_features':
            feature_vectors = zip(feature1, feature2, feature3, judgement_list)
        else:
            feature_vector_ks = zip(feature1, feature2, feature3, judgement_list)
            for i in range(3):
                all_tokens.append(0)
            for i, x in enumerate(feature_vector_ks):
                feature_vectors[i] += x
    
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
            if extractor is 'brown' or extractor is 'brown_full':
                brown_cluster_dict = get_brown_clusters()
                f.write("@ATTRIBUTE cluster_{} integer\n".format(i))
            elif extractor is not 'dependency_features' and extractor is not 'kitchen_sink':
                f.write("@ATTRIBUTE token_{} integer\n".format(i))

        if extractor is 'dependency_features':
            for i in range(3):
                f.write("@ATTRIBUTE feature_{} integer\n".format(i))

        if extractor is 'kitchen_sink':
            for i in range(len(all_tokens)):
                f.write("@ATTRIBUTE feature_{} integer\n".format(i))

        ### SPECIFY ADDITIONAL FEATURES HERE ###
        # For example: f.write("@ATTRIBUTE custom_1 REAL\n")
        #for i in range(len())
        #if extractor is 'dependency_features':
        #        f.write("@ATTRIBUTE feature2_{} integer\n".format(i))

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
    if extractor == 'bow':
        train_file = 'train_bow.arff'
        test_file = 'test_bow.arff'
    elif extractor == 'nltk_tokenizer':
	train_file = 'train_nltk.arff'
	test_file = 'test_nltk.arff'
    elif extractor == 'brown':
	train_file = 'train_brown.arff'
	test_file = 'test_brown.arff'
    elif extractor == 'brown_full':
	train_file = 'train_brown_full.arff'
	test_file = 'test_brown_full.arff'
    elif extractor == 'dependency_features':
	train_file = 'train_dependency.arff'
	test_file = 'test_dependency.arff'
    elif extractor == 'kitchen_sink':
	train_file = 'train_kitchen_sink.arff'
	test_file = 'test_kitchen_sink.arff'
    else:
        train_file = 'train.arff'
	test_file = 'test.arff'

    data, all_tokens = parse_data(TRAIN_DATA_PATH, TEST_DATA_PATH, extractor)
    feature_vectors = create_feature_vectors(data, all_tokens, extractor)
    generate_arff_file(feature_vectors[:6000], all_tokens, train_file, extractor)
    generate_arff_file(feature_vectors[6000:], all_tokens, test_file, extractor)
