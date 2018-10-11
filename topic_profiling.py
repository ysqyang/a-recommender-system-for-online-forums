from gensim import corpora, models
import collections
from sklearn import preprocessing
import stream
import numpy as np
import sys
import warnings

def get_scores(db, topic_id, features, weights, rid_to_index, tid_to_reply_table):
    '''
    Computes importance scores for replies under each topic
    Args:
    db:                 database
    topic_id:           integer identifier for a topic
    features:           attributes to include in importance evaluation
    weights:            weights associated with attributes in features
    rid_to_index:       mapping from reply id to corpus index
    tid_to_reply_table: mapping from topic id to replies table number
    Returns:
    importance scores for replies
    '''
    # normalize weights
    s, scores, scaler = sum(weights), {}, preprocessing.MinMaxScaler() 
    norm_weights = [wt/s for wt in weights]

    reply_table_num = tid_to_reply_table[topic_id]     
    attrs = ', '.join(['REPLYID']+features)
    sql = '''SELECT {} FROM replies_{}
             WHERE TOPICID = {}'''.format(attrs, reply_table_num, topic_id)
    with db.query(sql) as cursor:
        results = cursor.fetchall()
        if len(results) == 0:
            return scores
        # normalize features using min-max scaler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_norm = scaler.fit_transform(np.array(results)[..., 1:])
            for result, feature_vec in zip(results, features_norm):
                if result[0] in rid_to_index:
                    corpus_index = rid_to_index[result[0]]
                    scores[corpus_index] = np.dot(feature_vec, norm_weights)

    return scores

def get_word_weight(corpus_under_topic, dictionary, scores, alpha=0.7, 
                     smartirs='atn'):
    '''
    Computes word importance in a weighted corpus
    Args:
    corpus_under_topic: Corpus_under_topic object for a given topic 
    dictionary:         gensim dictionary object created from corpus
    scores:             list of reply scores under each topic
    alpha:              contribution coefficient for the topic content
    smartirs:           tf-idf weighting variants 
    Returns:
    dict of word importance values
    '''
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_under_topic]
    language_model = models.TfidfModel(corpus_bow, smartirs=smartirs)

    # if there is no replies under this topic, use augmentedterm frequency
    # as word weight
    if len(corpus_bow) == 1:
        if len(corpus_bow[0]) == 0:
            return {}
        max_freq = max(x[1] for x in corpus_bow[0])
        return {x[0]:0.5+0.5*x[1]/max_freq for x in corpus_bow[0]}

    word_weight = collections.defaultdict(float)
    # get the max score under each topic for normalization purposes
    max_score = max(scores)

    for i, doc in enumerate(corpus_bow):
        if len(doc) == 0:
            continue
        converted = language_model[doc]
        if len(converted) == 0:
            continue
        max_word_weight = max([x[1] for x in converted])
        coeff = 1-alpha if i else alpha
        score_norm = scores[i]/max_score if i else 1 
        for word_id, weight in converted:
            word_weight_norm = weight/max_word_weight
            word_weight[word_id] += coeff*score_norm*word_weight_norm

    return word_weight

def get_word_weight_all(db, tid_to_table, tid_to_reply_table, features, 
                        weights, preprocess_fn, stopwords, alpha=0.7, 
                        smartirs='atn'):

    '''
    Computes word weight dictionary for all discussion threads
    Args:
    db:                 database connection
    tid_to_table:       dictionary mapping topic id to topic table number
    tid_to_reply_table: dictionary mapping topic id to reply table number
    features:           attributes to include in importance evaluation
    weights:            weights associated with attributes in features
    preprocess_fn:      function to preprocess original text 
    stopwords:          set of stopwords
    alpha:              contribution coefficient for the topic content   
    smartirs:           tf-idf weighting variants
    Returns:
    Word importance values for all topics
    '''
    percentage, weight = .05, {}
    n_topic_ids = len(tid_to_table)
    print('Computing word weights for all topics...')
    # create a Corpus_under_topic object for each topic
    for i, topic_id in enumerate(tid_to_table):
        corpus = stream.Corpus_under_topic(db, topic_id, 
                                           tid_to_table[topic_id],
                                           tid_to_reply_table[topic_id], 
                                           preprocess_fn, stopwords)
        
        dictionary = corpora.Dictionary(corpus)
        
        scores = get_scores(db, topic_id, features, weights, 
                            corpus.reply_id_to_corpus_index) 

        weight[topic_id] = get_word_weight(corpus, dictionary, 
                                           scores, alpha, smartirs)

        if i+1 == int(n_topic_ids*percentage):
            print('{}% finished'.format(percentage))
            percentage += .05

    return weight

def get_top_k_words(word_weight, k):
    '''
    Get the top k most important words for a topic
    Args:
    word_weight: dictionary mapping words to weights
    k:           number of words to represent the topic
    Returns:
    Top k most important words for a topic
    '''
    if k > len(word_weight):
        k = len(word_weight)

    word_weight = [(w, word_weight[w]) for w in word_weight]
    
    word_weight.sort(key=lambda x:x[1], reverse=True)

    return [x[0] for x in word_weight[:k]] 
    








