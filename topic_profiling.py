from gensim import corpora, models, similarities
import jieba
import comment_scoring

class Corpus_for_streaming(object):
	def __init__(self, path):
		self.path = path

	def __iter__(self):
		with open(self.path, 'r') as f:
			yield f.readline()

def build_corpus(text_ids, fetch_text_fn, save_to_disk):
	'''
	Builds a corpus from texts, optionally saves the built 
	corpus to disk
	Args:
	text_ids:      list of text id's
	fetch_text_fn: function to fetch text from 
	save_ro_disk:  whether to save the built corpus to disk  
	'''
	if save_to_disk:
		with open('corpus.txt', 'w') as f:
			for id_ in text_ids:
				f.write(fetch_text_fn(id_))

		return None

	return [fetch_text_fn(id_) for id_ in text_ids]

def preprocess_corpus(raw_corpus, preprocess_fn, stopwords_dict):
	'''
	Preprocesses a corpus to a desired representation
	Args:
	raw_corpus:    raw_corpus file path (a string) or reference
	               to corpus (python list) stored in memory
	preprocess_fn: function to preprocess (e.g., tokenize and 
	               filter) raw corpus
	Returns:
	Preprocessed corpus stored in memory
	'''
	if type(raw_corpus) == str: # if corpus is a path
		corpus = []
		stream = Corpus_for_streaming(raw_corpus)
		for text in stream:
			corpus.append(preprocess_fn(text, stopwords_dict))
		return corpus
	
	return [preprocess_fn(text, stopwords_dict) for text in raw_corpus] 

def term_importance(corpus, weight_fn, model):
	'''
	Computes word importance in a corpus with weighted docs
	Args:
	texts:     list of text id's
	weight_fn: function to compute weight for each text
	model:     language model to convert corpus to a desirable 
	           represention
	'''
	dictionary = corpora.Dictionary(corpus)
	corpus = [dictionary.doc2bow(text) for text in corpus]
	language_model = model(corpus)
	converted = language_model[corpus]
	weights = weight_fn()

def 




