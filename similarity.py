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

def compute_word_probability(dictionary, word_freq_doc, word_freq_corpus, coeff):
    '''
    Computes word probabilities w.r.t. each document in the corpus
    Args:
    dictionary:          gensim dictionary object created from 
                         corpus_all_topics
    word_freq_in_docs:   list of word frequencies in each document
    word_freq_in_corpus: dictionary of word frequencies in the entire corpus
    coeff:               contribution coefficient for in-document word frequency
    Returns:
    List of word probabilities w.r.t. each document 
    '''
    word_prob_doc = []
    for word_freq in word_freq_doc:
        word_probs = {}
        # iterate through all words by id
        for word_id in dictionary.token2id.values():
            # contribution from word frequency in corpus 
            word_prob[word_id] = (1-coeff)*word_freq_corpus[word_id]
            # contribution from word frequency in document
            if word_id in word_freq:
                word_prob[word_id] += coeff*word_freq[word_id]
        word_prob_doc.append(copy.deepcopy(word_prob))

    return word_prob_doc

def compute_joint_probability(dictionary, profile_words, word_prob_doc):
    '''
    Computes the unnormialized joint probability of observing target_word
    and profile_words together in corpus
    Args:
    dictionary:    gensim dictionary object created from 
                   corpus_all_topics
    profile_words: list words that represent a discussion thread
                   (i.e., topic with all replies)
    word_prob_doc: word probabilities w.r.t. each document
    Returns:
    Unnormialized joint probability of observing each dictionary word 
    along with the profile words together in corpus
    '''
    tok2id = dictionary.token2id
    profile_word_ids = [tok2id(w) for w in profile_words]
    joint_prob = [0]*len(tok2id)
    # compute the join probability of observing each dictionary
    # word together with profile words 
    for word_id in dictionary.token2id.values():        
        prob = 1 if word_id in profile_word_ids else vec[word_id]
        # iterate through each document in corpus
        for vec in word_prob_doc:
            # compute the joint probability for each doc 
            for profile_word_id in profile_word_ids:
                prob *= vec[profile_word_id]

            # assuming uniform prior distribution over all docs in corpus,
            # the joint probability is the sum of joint probabilities over
            # all docs
            joint_prob[word_id] += prob

    return joint_prob 



'''

_CORPUS = './sample_corpus.txt'
_STOPWORDS = './stopwords.txt'
stopwords = run.load_stopwords(_STOPWORDS)

corpus = Corpus_all_topics(_CORPUS, stopwords, utilities.preprocess)

dictionary = corpora.Dictionary(corpus)

in_docs, in_corpus = compute_word_frequency(dictionary, corpus)

print(in_docs)
print(in_corpus)

probs = compute_word_probability(dictionary, in_docs, in_corpus, 0.5)

for vec in probs:
    print(sum(vec.values()))






