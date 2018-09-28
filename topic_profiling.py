from gensim import corpora, models, similarities
import jieba
import comment_scoring
import random
import collections
import pandas as pd
from sklearn import preprocessing
import numpy as np
import csv

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


def build_corpus(in_file_path, out_file_path):
    '''
    Builds raw corpus from a csv file containing the original database
    Args: 
    in_file_path:  path for input file
    out_file_path: path for output file   
    Returns:
    path for raw corpus file 
    '''
    index_to_textid = {}
    d = collections.defaultdict(int)
    with open(in_file_path, 'r') as in_file, open(out_file_path, 'w') as out_file
        cnt = 0
        reader = csv.reader(in_file)
        writer = csv.writer(out_file)
        for line in reader:
            d[len(line)] += 1
            index_to_text_id[int(line[''])] = cnt
            writer.writeline(line[3])
            cnt += 1

    print(d)

    return out_file_path

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

'''
corpus_path, stopwords_path = './corpus.txt', './stopwords.txt'
dictionary = build_dictionary(corpus_path, comment_scoring.preprocess, stopwords_path)
'''
def compute_scores(df, features):
    '''
    Computes scores for each text
    Args:
    df:       Pandas dataframe object containing original database
    features: list of attributes to include in computing scores
    weights:  list of weights for the attributes in features
    '''
    # normalize weights
    norm_weights = [wt/sum(weights) for wt in weights]

    for feature in features:
        print(feature, max(df[feature]), min(df[feature]))

    # normalize features using min-max-scaler
    scaler = preprocessing.MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])


    scores = df.apply(lambda x:np.dot(x[features], norm_weights), axis=1)

    return scores.to_dict()

def create_data_frame(path, error_bad_lines):
    return pd.read_csv(path, error_bad_lines=error_bad_lines)

df = create_data_frame('./topics_0.csv', False)
features = ['GOLDUSEFULNUM', 'USEFULNUM', 'TOTALPCPOINT', 'TOPICPCPOINT', 'TOTALVIEWNUM', 'TOTALREPLYNUM']
weights=[1]*6


scores = compute_scores(df, features)

print(scores)