from gensim import corpora, models
import collections
from sklearn import preprocessing
import stream

def get_scores(db, topic_id, features, weights, id_to_index):
    '''
    Computes importance scores for replies under each topic
    Args:
    db:          pymysql database connection 
    topic_id:    integer identifier for a topic
    features:    attributes to include in importance evaluation
    weights:     weights associated with attributes in features
    id_to_index: mapping from reply id to corpus index
    Returns:
    importance scores for replies
    '''
    # normalize weights
    s, scores, scaler = sum(weights), {}, preprocessing.MinMaxScaler() 
    norm_weights = [wt/s for wt in weights]

    for i in range(10):       
        with db.cursor() as cursor:
            attrs = ', '.join(['REPLYID']+features)
            sql = '''SELECT {} FROM replies_{}
                     WHERE TOPICID = {}'''.format(attrs, i, topic_id)
            cursor.execute(sql)
            results = cursor.fetchall()
            # normalize features using min-max scaler
            features_norm = scaler.fit_transform(np.array(results)[..., 1:])

            for result, feature_vec in zip(results, features_norm):
                corpus_index = id_to_index[result[0]]
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
    word_weight = collections.defaultdict(float)

    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_under_topic]
    language_model = models.TfidfModel(corpus_bow, smartirs=smartirs)

    # get the max score under each topic for normalization purposes
    max_score = max(scores)
    for i, doc in enumerate(corpus_bow):
        converted = language_model[doc]
        max_word_weight = max([x[1] for x in converted])
        coeff, score_norm = 1-alpha, scores[i]/max_score if i else alpha, 1 
        for word in converted:
            word_weight_norm = word[1]/max_word_weight
            word_weight[word[0]] += coeff*score_norm*word_weight_norm

    return word_weight

def get_word_weight_all(db, tid_to_table, features, weights, preprocess_fn, 
                         stopwords, alpha=0.7, smartirs='atn'):

    '''
    Computes word weight dictionary for all discussion threads
    Args:
    db:            database connection
    tid_to_table:  dictioanry mapping topic id's to table numbers
    features:      attributes to include in importance evaluation
    weights:       weights associated with attributes in features
    preprocess_fn: function to preprocess original text 
    stopwords:     set of stopwords
    alpha:         contribution coefficient for the topic content   
    smartirs:           tf-idf weighting variants
    Returns:
    Word importance values for all topics
    '''
    weight = {}

    # create a Corpus_under_topic object for each topic
    for topic_id in tid_to_table:
        corpus = stream.Corpus_under_topic(db, topic_id, 
                                           tid_to_table[topic_id], 
                                           preprocess_fn, stopwords)
        
        dictionary = corpora.Dictionary(corpus)
        
        scores = get_scores(db, topic_id, features, weights, 
                            corpus.reply_id_to_corpus_index)        
        
        weight[topic_id] = get_word_weight(corpus, dictionary, 
                                            scores, alpha, smartirs)

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
    








