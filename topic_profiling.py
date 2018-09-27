from gensim import corpora, models, similarities
import jieba
import comment_scoring
import random
import collections
import pandas

class Corpus_stream(object):
    '''
    Corpus object for streaming preprocessed texts
    '''
    def __init__(self, corpus_path, stopwords_path, preprocess_fn, dictionary):
        self.corpus_path = corpus_path
        self.stopwords_path = stopwords_path
        self.preprocess_fn = preprocess_fn
        self.dictionary = dictionary

    def __iter__(self):
        with open(self.corpus_path, 'r') as f:
            while True:
                raw_text = f.readline().strip()
                if raw_text == '':
                    break
                yield self.dictionary.doc2bow(
                    self.preprocess_fn(raw_text, self.stopwords_path))

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

def build_dictionary(corpus_path, preprocess_fn, stopwords_path):
    return corpora.Dictionary(preprocess_fn(line.rstrip(), stopwords_path) 
                              for line in open(corpus_path, 'r')) 

def word_importance(corpus_path, stopwords_path, preprocess_fn, 
                    dictionary, model, normalize):
    '''
    Computes word importance in a weighted corpus
    Args:
    corpus_path:    raw text file path
    stopwords_path: stopword file path
    preprocess_fn:  function to preprocess raw text
    dictionary:     gensim Dictionary object
    model:          language model to convert corpus to a desirable 
                    represention (e.g., tf-idf)
    normalize:      whether to normalize the representation obtained from model
    Returns:
    dict of word importance values
    '''
    for id_ in dictionary:
        print(id_, dictionary[id_])
    stream = Corpus_stream(corpus_path, stopwords_path, preprocess_fn, dictionary)
    language_model = model(stream, normalize=normalize)

    word_weights = collections.defaultdict(float)
    for text in stream:
        converted = language_model[text]
        max_word_weight = max([x[1] for x in converted])
        for word in converted:
            word_weight_norm = word[1]/max_word_weight
            word_weights[word[0]] += word_weight_norm

    return word_weights


corpus_path, stopwords_path = './corpus.txt', './stopwords.txt'
dictionary = build_dictionary(corpus_path, comment_scoring.preprocess, stopwords_path)

word_weights = word_importance(corpus_path, stopwords_path, comment_scoring.preprocess, dictionary, models.TfidfModel, False)

print(word_weights)


