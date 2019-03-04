# data loading
import numpy as np
import pandas as pd
import math

# preprocessing (paragragh vectors)
import gensim
import multiprocessing
from functools import reduce

# MLP Classifier
from sklearn.neural_network import MLPClassifier

# Evaluation
from sklearn.metrics import confusion_matrix

def load_data(datasets):
	'''Reads in and formats data from the list of datasets given.

	Args:
		datasets: A list of Dataset objects
	
	Returns:
		A numpy array that is the concatenation of the numpy arrays for all the
		given datasets.
	'''
	return np.concatenate([_load_data(dataset) for dataset in datasets])

def _load_data(dataset):
	'''Helper function to load_data. Reads in file for a single dataset given.
	
	Args:
		dataset: A Dataset object
	
	Returns:
		A numpy array with only the required columns from dataset.
	'''
	df = pd.read_csv(dataset.path, sep=',', encoding='ISO-8859-1')
	raw_data = np.array(df)

	# get the columns for Subject and Severity Rating
	extract_cols = [1, 2]
	del_cols = np.delete(np.arange(raw_data.shape[1]), extract_cols)
	data = np.delete(raw_data, del_cols, axis=1)

	# check for possible NaN severity values
	del_rows = []
	for i in range(len(data)):
		if math.isnan(data[i][1]):
			del_rows.append(i)

	# delete rows that contain NaN severity values
	if len(del_rows) > 0:
		data = np.delete(data, del_rows, axis=0)

	# add column for project id
	dataset_size = len(data)
	project_id_column = [dataset.project_id for i in range(dataset_size)]
	data = np.insert(data, 2, project_id_column, axis=1)

	return data

#def __data(data, percent):
	'''Split 
	'''


def preprocess(train_data, test_data):
	# construct testing and training corpora (plural or corpus)
	train_corpus = list(_read_corpus(train_data))
	test_corpus = list(_read_corpus(test_data, tokens_only=True))

	cores = multiprocessing.cpu_count()

	# instantiate Doc2Vec object
	# !! switch to epoch 4000 or 40000 when done testing 
	model_DM = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=2, epochs=400, workers=cores,  dm=1, dm_concat=1 )
	model_DBOW = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=2, epochs=400, workers=cores, dm=0)

	# build a vocabulary
	model_DM.build_vocab(train_corpus)
	model_DBOW.build_vocab(train_corpus)

	# train Doc2Vec
	model_DM.train(train_corpus, total_examples=model_DM.corpus_count, epochs=model_DM.epochs)
	model_DBOW.train(train_corpus, total_examples=model_DBOW.corpus_count, epochs=model_DBOW.epochs)

	# create test and train sets using Doc2Vec output
	#X_train = [(list(model_DM.docvecs[i]) + list(model_DBOW.docvecs[i])) for i in range(len(train_data))]
	X_train = []
	for i in range(len(train_data)):
		doc_vec = ((list(model_DM.docvecs[i]) + list(model_DBOW.docvecs[i])))
		#project_vec = [reduce((lambda x, y: x + y), doc_vec), train_data[i][2]]
		doc_vec.append(train_data[i][2])
		X_train.append(doc_vec)
	Y_train = [doc[1] for doc in train_data]

	#X_test = [(list(model_DM.infer_vector(test_corpus[i])) + list(model_DBOW.infer_vector(test_corpus[i]))) for i in range(len(test_data))]
	X_test = []
	for i in range(len(test_data)):
		doc_vec = (list(model_DM.infer_vector(test_corpus[i])) + list(model_DBOW.infer_vector(test_corpus[i])))
		#project_vec = [reduce((lambda x, y: x + y), doc_vec), test_data[i][2]]
		doc_vec.append(test_data[i][2])
		X_test.append(doc_vec)
	Y_test = [doc[1] for doc in test_data]

	return X_train, Y_train, X_test, Y_test


def _read_corpus(data, tokens_only=False):
	for i, line in enumerate(data):
		if tokens_only:
			yield gensim.utils.simple_preprocess(line[0])
		else:
			# For training data, add tags
			yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line[0]), [i])

			
class ASP():
	def __init__(self, X_train, Y_train, X_test, Y_test):
		self.X_train, self.Y_train = X_train, Y_train
		self.X_test, self.Y_test = X_test, Y_test
		self.classifier = MLPClassifier(alpha = 0.7, max_iter=10000)
		
	def fit(self):
		self.classifier.fit(self.X_train, self.Y_train)

		df_results = pd.DataFrame(data=np.zeros(shape=(0,3)), columns = ['classifier', 'train_score', 'test_score'] )
		train_score = self.classifier.score(X_train, Y_train)
		test_score = self.classifier.score(X_test, Y_test)
		 
		#print  (classifier.predict_proba(X_test))
		#print  (classifier.predict(X_test))
		 
		df_results.loc[1,'classifier'] = "MLP"
		df_results.loc[1,'train_score'] = train_score
		df_results.loc[1,'test_score'] = test_score
		print(df_results)
		

	def predict(self):
		prediction = self.classifier.predict(self.X_test)
		matrix = confusion_matrix(Y_test, prediction, labels=[1, 2, 3, 4])

		print(matrix)

class Dataset():
	def __init__(self, path, project_id):
		self.path = path
		self.project_id = project_id

# testing code -----------------------------------------------------------------
# list of dataset objects
pits_train = [Dataset('../dataset/raw/pitsA.csv', 1),
              Dataset('../dataset/raw/pitsB.csv', 2),
              Dataset('../dataset/raw/pitsC.csv', 3),
              Dataset('../dataset/raw/pitsD.csv', 4),
              Dataset('../dataset/raw/pitsE.csv', 5)]

pits_test = [Dataset('../dataset/raw/pitsF.csv', 6)]

# load data
train_data = load_data(pits_train)
test_data = load_data(pits_test)
print('loaded data')

# create training and testing sets
X_train, Y_train, X_test, Y_test = preprocess(train_data, test_data)
print('preprocessed')

# classify
classifier = ASP(X_train, Y_train, X_test, Y_test)
classifier.fit()
print('trained classifier')

classifier.predict()
print('prediction complete')