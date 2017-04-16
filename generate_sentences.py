TEST_DATA_PATH = "test.tsv"
TRAIN_DATA_PATH = "train.tsv"
OUTPUT_FILE_PATH = "sentence.txt"

if __name__ == "__main__":
	with open(TRAIN_DATA_PATH) as inputted_file:
		train_data_lines = inputted_file.read().splitlines()
		
	with open(TEST_DATA_PATH) as inputted_file:
		test_data_lines = inputted_file.read().splitlines()
		
	write_output = open(OUTPUT_FILE_PATH, 'w')

	for line in train_data_lines:
		sentences = line.split('\t')
		sentence = sentences[3]
		words = sentence.split()
		write_output.write("%s.\n" % sentence)
		
	for line in test_data_lines:
		sentences = line.split('\t')
		sentence = sentences[3]
		write_output.write("%s.\n" % sentence)
		
	write_output.close()
