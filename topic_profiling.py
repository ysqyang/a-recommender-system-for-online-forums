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

def preprocess_corpus(corpus, preprocess_fn):
	'''
	Preprocesses a corpus to a desired representation
	Args:
	corpus:        corpus file path (a string) or reference
	               to corpus (python list) stored in memory
	preprocess_fn: function to preprocess corpus
	'''
	if type(corpus) == str:
		with open(corpus, 'rw') as f:
			



def term_importance(text_ids, fetch_text_fn, tokenize_fn, 
	                weight_fn, model, stopwords_dict):
	'''
	Computes word importance in a corpus with weighted docs
	Args:
	texts:     list of text id's
	weight_fn: function to compute weight for each text
	model:     language model 
	'''
	corpus = []
	for id_, text in texts.items():
		corpus.append(tokenize_fn(text, stopwords_dict))

	dictionary = corpora.Dictionary(corpus)
	corpus = [dictionary.doc2bow(text) for text in corpus]

	language_model = model(corpus)
	converted = tfidf[corpus]

	for doc in converted:
	    print(doc)




