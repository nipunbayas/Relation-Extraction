#import weka.core.jvm as jvm
#jvm.start()

#from weka.core.converters import Loader
from __future__ import division
from weka.classifiers import Classifier

input_file = 'test.tsv'
gold_standard_result = dict()
# Use extractor type as either - 'normal', 'nltk_tokenizer', 'brown', 'brown_full', 'dependency_features' or 'kitchen_sink'
extractor = 'nltk_tokenizer'

input_file = 'test.tsv'
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

if extractor == 'normal':
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
	

with open(input_file) as inputted_file:
        lines = inputted_file.read().splitlines()  # read input file
        
        
#loader = Loader(classname="weka.core.converters.ArffLoader")
#data = loader.load_file("train.arff")
#data.class_is_last()

#print(data)
#out_path = 'bow_model.txt'
cls = Classifier(name='weka.classifiers.functions.LibSVM', ckargs={'-C':0.1})
#cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
#cls.build_classifier(data)

#for index, inst in enumerate(data):
#    pred = cls.classify_instance(inst)
#    dist = cls.distribution_for_instance(inst)
#    print(str(index+1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))

cls.train(train_file)
predictions = cls.predict(test_file)

count = 1
for line in lines:
        sentences = line.split('\t')
        gold_standard_result[count] = sentences[-1]
        count += 1

count = 1
for result in predictions:
	print result.predicted, gold_standard_result[count]
	if result.predicted == 'yes' and gold_standard_result[count] == 'yes':
		true_positives += 1
	elif result.predicted == 'yes' and gold_standard_result[count] == 'no':
		false_positives += 1
	elif result.predicted == 'no' and gold_standard_result[count] == 'yes':
		false_negatives += 1
	elif result.predicted == 'no' and gold_standard_result[count] == 'no':
		true_negatives += 1
	count += 1

print true_positives, ", ", true_negatives, ", ", false_positives, ", ", false_negatives
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * ((precision * recall) / (precision + recall))
print precision, recall, f1_score

#jvm.stop()
