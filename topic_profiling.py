from gensim import corpora, models, similarities
import jieba
import comment_scoring
import random
import collections

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
    Returns:
    path to corpus file if save_to_disk is true or raw_corpus 
    if save_to_disk is false 
    '''
    if save_to_disk:
        with open('./corpus.txt', 'w') as f:
            for id_ in text_ids:
                f.write(fetch_text_fn(id_))

        return 

    return [fetch_text_fn(id_) for id_ in text_ids]

def preprocess_corpus(raw_corpus, preprocess_fn, stopwords_set):
    '''
    Preprocesses a corpus to a desired representation
    Args:
    raw_corpus:     raw_corpus file path (a string) or reference
                    to corpus (python list) stored in memory
    preprocess_fn:  function to preprocess (e.g., tokenize and 
                    filter) raw corpus
    stopwords_dict: path of stopword dictionary for use in preprocessing
    Returns:
    Preprocessed corpus stored in memory
    '''
    if type(raw_corpus) == str: # if corpus is a path
        corpus = []
        stream = Corpus_for_streaming(raw_corpus)
        for text in stream:
            corpus.append(preprocess_fn(text, stopwords_set))
        return corpus
    
    return [preprocess_fn(text, stopwords_set) for text in raw_corpus] 

def get_weights():
    

def word_importance(corpus, weights, model):
    '''
    Computes word importance in a weighted corpus
    Args:
    corpus:    corpus containing weighted texts
    weights:   list of text weights
    model:     language model to convert corpus to a desirable 
               represention (e.g., tf-idf)
    Returns:
    dict of word importance values
    '''
    dictionary = corpora.Dictionary(corpus)
    for id_ in dictionary:
        print(id_, dictionary[id_])

    corpus = [dictionary.doc2bow(text) for text in corpus]
    language_model = model(corpus)
    converted = language_model[corpus]
    max_text_weight = max(weights)

    word_weights = collections.defaultdict(float)
    for text, text_weight in zip(converted, weights):
        print('text,', text, 'weight, ', text_weight)
        max_word_weight = max([x[1] for x in text])
        text_weight_norm = text_weight/max_text_weight
        for word in text:
            word_weight_norm = word[1]/max_word_weight
            word_weights[dictionary[word[0]]] += text_weight_norm*word_weight_norm

    return word_weights

'''
raw_corpus = ['我昨天去上海了',
          '今天天气好热，不过下周的天气会很舒服',
          '他们在犹豫晚上要不要出去吃饭，后来还是在家里吃饭了',
          '你后天要去北京，北京这几天雾霾很厉害',
          '她儿子考上了清华大学',
          '最近股票跌得很厉害',
          '今年高考她没考好',
          '这两年经济不景气啊',
          '下周要下雪了'
          ]

weights = [random.random() for _ in range(len(raw_corpus))]
print(weights)

stopwords_set = comment_scoring.get_stopwords('./stopwords.txt')
corpus = preprocess_corpus(raw_corpus, comment_scoring.tokenize, stopwords_set)
print(corpus)

word_weights = word_importance(corpus, weights, models.TfidfModel)
print(word_weights)
'''



