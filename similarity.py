from gensim import corpora
import utilities
import copy
import collections
from pprint import pprint
# import stream

'''
_TEXT = ['我昨天去上海了',
         '今天天气好热，不过下周的天气会很舒服',
         '他们在犹豫晚上要不要出去吃饭，后来还是在家里吃饭了',
         '你后天要去北京，北京这几天雾霾很厉害',
         '她儿子考上了清华大学',
         '最近股票跌得很厉害',
         '今年高考她没考好',
         '这两年经济不景气啊',
         '下周要下雪了'
         ]

'''
class Corpus_all_topics(object):
    '''
    Corpus object for streaming and preprocessing 
    texts from topics_info tables
    '''
    def __init__(self, path, stopwords, preprocess_fn):
        self.path = path
        self.stopwords = stopwords 
        self.preprocess_fn = preprocess_fn
        
    def __iter__(self):
        # iterates through all topics
        with open(self.path, 'r') as f:
        	while True:
        		doc = f.readline().strip()
        		if doc == '':
        			break
        		yield self.preprocess_fn(doc, self.stopwords)

def compute_word_frequency(dictionary, corpus_all_topics):
	'''
	Computes normalized word frequencies in each document in 
	corpus_all_topics and in the entire corpus_all_topics
	Args:
	dictionary:        gensim dictionary object created from 
	                   corpus_all_topics
	corpus_all_topics: corpus containing the contents of all topics
	Returns:
	List of word frequencies in each document and dictionary
	of word frequencies in the entire corpus 
	'''
	word_freq_in_docs = []
	word_freq_in_corpus = collections.defaultdict(int)
	bow_corpus = [dictionary.doc2bow(doc) for doc in corpus_all_topics]
	num_tokens_in_corpus = sum(sum(x[1] for x in vec) for vec in bow_corpus)
	for vec in bow_corpus:
		word_freq = {}
		num_tokens = sum(x[1] for x in vec)
		for (word_id, count) in vec:
			word_freq[word_id] = count/num_tokens
			word_freq_in_corpus[word_id] += count
		word_freq_in_docs.append(copy.deepcopy(word_freq))
		
	for word_id in word_freq_in_corpus:
		word_freq_in_corpus[word_id] /= num_tokens_in_corpus

	return word_freq_in_docs, word_freq_in_corpus

def compute_word_probability(dictionary, word_freq_in_docs, word_freq_in_corpus, coeff):
	'''
	Computes word probabilities w.r.t. each document in the corpus
	Args:
	dictionary:          gensim dictionary object created from 
	                     corpus_all_topics
	word_freq_in_docs:   list of word frequencies in each document
	word_freq_in_corpus: dictionary of word frequencies in the entire corpus 
	Returns:
	List of word probabilities 
	'''
	word_probs_in_docs = []
	for word_freq in word_freq_in_docs:
		word_probs = {}
		for word_id in dictionary.token2id.values():
			word_probs[word_id] = (1-coeff)*word_freq_in_corpus[word_id]
			if word_id in word_freq:
				word_probs[word_id] += coeff*word_freq[word_id]
		word_probs_in_docs.append(copy.deepcopy(word_probs))

	return word_probs_in_docs


def 
'''
_CORPUS = './sample_corpus.txt'
_STOPWORDS = './stopwords.txt'
stopwords = utilities.load_stopwords(_STOPWORDS)

corpus = Corpus_all_topics(_CORPUS, stopwords, utilities.preprocess)

dictionary = corpora.Dictionary(corpus)

in_docs, in_corpus = compute_word_frequency(dictionary, corpus)

print(in_docs)
print(in_corpus)

probs = compute_word_probability(dictionary, in_docs, in_corpus, 0.5)

pprint(probs)
'''





