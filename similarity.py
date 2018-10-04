from gensim import corpora
import utilities
import copy
import numpy as np
from scipy import stats
import collections
import math
from pprint import pprint
# import stream

class Corpus_all_topics(object):
    '''
    Corpus object for streaming and preprocessing 
    texts from topics_info tables
    '''
    def __init__(self, path, preprocess_fn, stopwords):
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

def get_word_frequency(dictionary, corpus_all_topics):
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
    word_freq_doc = []
    word_freq_corpus = collections.defaultdict(int)
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus_all_topics]
    num_tokens_corpus = sum(sum(x[1] for x in vec) for vec in bow_corpus)
    # iterate through documents in corpus
    for vec in bow_corpus:
        word_freq = {}
        # total number of tokens (with repetitions) in current doc 
        num_tokens = sum(x[1] for x in vec)
        for (word_id, count) in vec:
            # normalized word frequency in current doc
            word_freq[word_id] = count/num_tokens
            # update word frequency in corpus 
            word_freq_corpus[word_id] += count

        # add word frequency vector for current doc to result list
        word_freq_doc.append(copy.deepcopy(word_freq))
        
    for word_id in word_freq_corpus:
        word_freq_corpus[word_id] /= num_tokens_corpus

    return word_freq_doc, word_freq_corpus

def get_word_doc_prob(tok2id, word_freq_doc, word_freq_corpus, coeff):
    '''
    Computes the word probabilities w.r.t. each document in the corpus
    Args:
    tok2id:              token-to-id mapping 
    word_freq_in_docs:   list of word frequencies in each document
    word_freq_in_corpus: dictionary of word frequencies in the entire corpus
    coeff:               contribution coefficient for in-document word frequency
                         in computing word frequency in document
    Returns:
    List of word probabilities w.r.t. each document 
    '''
    word_prob_doc = []
    for word_freq in word_freq_doc:
        # contribution from word frequency in corpus
        prob = [(1-coeff)*word_freq_corpus[wid] for wid in tok2id.values()]
        # iterate through all words by id
        for word_id in word_freq:
            # contribution from word frequency in document
            prob[word_id] += coeff*word_freq[word_id]
        word_prob_doc.append(copy.deepcopy(prob))

    return word_prob_doc

def get_prob_topic_profile(tok2id, profile_word_ids, word_prob_doc):
    '''
    Computes the conditional probability of observing a word 
    given the apprearance of profile_words
    Args:
    tok2id:            token-to-id mapping 
    profile_words_ids: list of word id's that represent a discussion thread
                       (i.e., topic with all replies)
    word_prob_doc:     word probabilities w.r.t. each document
    Returns:
    Conditional probability of observing a word given the apprearance of profile_words
    '''
    prob = [0]*len(tok2id)
    # compute the join probability of observing each dictionary
    # word together with profile words 
    for word_id in tok2id.values():        
        # iterate through each document in corpus
        for vec in word_prob_doc:
            # compute the joint probability for each doc
            # convert to natural log to avoid numerical issues
            log_prob = 0 if word_id in profile_word_ids else math.log(vec[word_id])
            for profile_word_id in profile_word_ids:
                log_prob += math.log(vec[profile_word_id])
            # assuming uniform prior distribution over all docs in corpus,
            # the joint probability is the sum of joint probabilities over
            # all docs
            prob[word_id] += math.exp(log_prob)

    # normalize the probabilities
    s = sum(prob)
    for i in range(len(prob)):
        prob[i] /= s
    
    return prob

def get_similarity(prob_topic_profile, word_prob_doc):
    '''
    Computes the similarity scores between a topic profile and 
    the documents in the corpus
    Args:
    prob_topic_profile: word probabilities given a topic profile
    word_prob_doc:      word probabilities w.r.t. each document
    Returns:
    Similarity scores between a topic profile and the documents 
    in the corpus
    '''
    similarities = []
    for vec in word_prob_doc:
        similarities.append(stats.entropy(pk=prob_topic_profile, qk=vec))

    return similarities

def get_similarity_all(db, preprocess_fn, stopwords, profile_words, coeff):
    '''
    Computes the similarity scores between a topic profile and 
    each documents in the corpus
    Args:
    db:            database connection
    preprocess_fn: function to preprocess original text 
    stopwords:     set of stopwords
    profile_words: words representing a topic
    coeff:         contribution coefficient for in-document word frequency  
                   in computing word frequency in document
    Returns:
    Similarity scores between a topic profile and each documents in the corpus
    '''
    corpus = Corpus_all_topics(db, preprocess_fn, stopwords)
    dictionary = corpora.Dictionary(corpus)
    tok2id = dictionary.token2id
    freq_doc, freq_corpus = get_word_frequency(dictionary, corpus)
    prob_doc = get_word_doc_prob(tok2id, freq_doc, freq_corpus, coeff)

    similarity_all = {}
    for topic_id, words in profile_words.items():
        profile_word_ids = [tok2id[word] for word in words]
        prob = get_prob_topic_profile(tok2id, profile_word_ids, prob_doc)
        similarity_all[topic_id] = get_similarity(prob, prob_doc)

    return similarity_all


_CORPUS = './sample_corpus.txt'
_STOPWORDS = './stopwords.txt'
stopwords = utilities.load_stopwords(_STOPWORDS)

profile_words = {0:['雾', '霾'], 1:['股票']}

print(get_similarity_all(_CORPUS, utilities.preprocess, stopwords, profile_words, 0.5))



