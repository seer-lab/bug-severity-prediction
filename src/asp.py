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
from sklearn.metrics import precision_recall_fscore_support

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

	# filter dataset percent
	if dataset.percent != 1:
		size = len(data)
		train_size = int(size * dataset.percent)
		if dataset.train:
			# delete row not needed from last row
			data = np.delete(data, slice(train_size, size), axis=0)

			#print('Range to delete: ', train_size, '-', size)
		else:
			# delete rows not needed from first row
			data = np.delete(data, slice(0, train_size), axis=0)

			#print('Range to delete: ', 0, '-', train_size)

		#print('Size of Dataset: ', size)
		#print('Size of remaining', len(data))

	return data


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

		#df_results = pd.DataFrame(data=np.zeros(shape=(0,3)), columns = ['classifier', 'train_score', 'test_score'] )
		#train_score = self.classifier.score(self.X_train, self.Y_train)
		#test_score = self.classifier.score(self.X_test, self.Y_test)
		 
		#print  (classifier.predict_proba(X_test))
		#print  (classifier.predict(X_test))
		 
		#df_results.loc[1,'classifier'] = "MLP"
		#df_results.loc[1,'train_score'] = train_score
		#df_results.loc[1,'test_score'] = test_score
		#print(df_results)
		

	def predict(self):
		prediction = self.classifier.predict(self.X_test)

		#matrix = confusion_matrix(self.Y_test, prediction, labels=[1, 2, 3, 4])
		#print(matrix)

		prf = precision_recall_fscore_support(y_true=self.Y_test, y_pred=prediction, average='weighted')
		print('Precision | Recall | F-Score')
		print(prf)


class Dataset():
	def __init__(self, path, project_id, percent=1, train=True):
		self.path = path
		self.project_id = project_id
		self.percent = percent
		self.train = train

class Experiment():
	def __init__(self, train, test):
		self.train = train
		self.test = test

	def run(self):
		# load data
		train_data = load_data(self.train)
		test_data = load_data(self.test)
		#print('loaded data')

		# create training and testing sets
		X_train, Y_train, X_test, Y_test = preprocess(train_data, test_data)
		#print('preprocessed')

		# classify
		classifier = ASP(X_train, Y_train, X_test, Y_test)
		classifier.fit()
		#print('trained classifier')

		classifier.predict()
		#print('prediction complete')		

# SCRIPT START -----------------------------------------------------------------

# dataset variables
a = '../dataset/raw/pitsA.csv'
b = '../dataset/raw/pitsB.csv'
c = '../dataset/raw/pitsC.csv'
d = '../dataset/raw/pitsD.csv'
e = '../dataset/raw/pitsE.csv'
f = '../dataset/raw/pitsE.csv'

# ------------------------------------------------------------------------------
# pitsF experiments
# ------------------------------------------------------------------------------
#percent of test data that will be in training data
import time
print(time.ctime(time.time()))
start = time.time()

percent_range = [1, 0.20, 0.50, 0.80, 0.95]
experiment_count = 1
for per in percent_range:
	train_datasets = [
		[a, b, c, d, e],
		[a, b, c, d],
		[a, b, c],
		[a, b],
		[a],
		[b, c, d, e],
		[b, c, d],
		[b, c],
		[b],
		[c, d, e],
		[c, d],
		[c],
		[d, e],
		[d],
		[e]
	]

	for datasets in train_datasets:
		# set up training data and test data for experiment
		project_id = 1
		pits_train = []
		for dataset in datasets:
			pits_train.append(Dataset(dataset, project_id, percent=1, train=True))
			project_id += 1
		if per != 1:
			pits_train.append(Dataset(f, project_id, percent=per, train=True))
		pits_test = [Dataset(f, project_id, percent=per, train=False)]

		# run experiment
		print('-----EXPERIMENT ', experiment_count, ' START-----')
		print('Percent: ', per)
		print('Training Data: ', datasets)
		experiment = Experiment(pits_train, pits_test)
		experiment.run()
		print('-----EXPERIMENT ', experiment_count, ' END-------')
		experiment_count +=1
		print('')

print(time.ctime(time.time()))
print('TOTAL RUNTIME: ', time.time()-start, 's')
print('')
