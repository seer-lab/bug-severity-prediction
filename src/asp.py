# data loading
import numpy as np
import pandas as pd
import math

# preprocessing (paragragh vectors)
import gensim
import multiprocessing

# MLP Classifier
from sklearn.neural_network import MLPClassifier

def load_data(paths):
	# account for filename(s) as list or string
	if type(paths) is list:
		return np.concatenate([_load_data(path) for path in paths])
	else:
		return _load_data(paths)

def _load_data(path):
    df = pd.read_csv(path, sep=',', encoding='ISO-8859-1')
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
    
    return data

def preprocess(train_data, test_data):
	# construct testing and training corpora (plural or corpus)
	train_corpus = list(_read_corpus(train_data))
	test_corpus = list(_read_corpus(test_data, tokens_only=True))

	cores = multiprocessing.cpu_count()

	# instantiate Doc2Vec object
	model_DM = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40, workers=cores,  dm=1, dm_concat=1 )
	model_DBOW = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40, workers=cores, dm=0)

	# build a vocabulary
	model_DM.build_vocab(train_corpus)
	model_DBOW.build_vocab(train_corpus)

	# train Doc2Vec
	model_DM.train(train_corpus, total_examples=model_DM.corpus_count, epochs=model_DM.epochs)
	model_DBOW.train(train_corpus, total_examples=model_DBOW.corpus_count, epochs=model_DBOW.epochs)

	# create test and train sets using Doc2Vec output
	X_train = [(list(model_DM.docvecs[i]) + list(model_DBOW.docvecs[i])) for i in range(len(train_data))]
	Y_train = [doc[1] for doc in train_data]

	X_test = [(list(model_DM.infer_vector(test_corpus[i])) + list(model_DBOW.infer_vector(test_corpus[i]))) for i in range(len(test_data))]
	Y_test = [doc[1] for doc in test_data]

	return X_train, Y_train, X_test, Y_test


def _read_corpus(data, tokens_only=False):
	for i, line in enumerate(data):
		if tokens_only:
			yield gensim.utils.simple_preprocess(line[0])
		else:
			# For training data, add tags
			yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line[0]), [i])
			



