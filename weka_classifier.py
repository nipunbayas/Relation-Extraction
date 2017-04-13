#import weka.core.jvm as jvm
#jvm.start()

#from weka.core.converters import Loader
from weka.classifiers import Classifier

#loader = Loader(classname="weka.core.converters.ArffLoader")
#data = loader.load_file("train.arff")
#data.class_is_last()

#print(data)
out_path = 'bow_model.txt'
cls = Classifier(name='weka.classifiers.functions.LibSVM', ckargs={'-C':1})
#cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
#cls.build_classifier(data)

#for index, inst in enumerate(data):
#    pred = cls.classify_instance(inst)
#    dist = cls.distribution_for_instance(inst)
#    print(str(index+1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))

cls.train('train.arff')
predictions = cls.predict('test.arff', verbose=1)

for result in predictions:
	pass
#	print "Result is: ", result

#jvm.stop()
