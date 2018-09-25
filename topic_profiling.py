from gensim import corpora, models, similarities
import jieba
import comment_scoring

class Corpus_for_streaming(object):
	def __iter__(self):
		with open('mycorpus.txt') as f:
			yield dictionary.doc2bow(line.lower().split())

def term_importance(texts, fetch_text_fn, tokenize_fn, 
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




