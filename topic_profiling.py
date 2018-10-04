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
    s = sum(weights)
    norm_weights = [wt/s for wt in weights]
    
    scores = {}
    for i in range(10):       
        with db.cursor() as cursor:
            attrs = ', '.join(['REPLYID']+features)
            sql = '''SELECT {} FROM replies_{}
                     WHERE TOPICID = {}'''.format(attrs, i, topic_id)
            cursor.execute(sql)
            results = cursor.fetchall()
            # normalize features using min-max scaler
            features_norm = scaler.fit_transform(np.array(results)[..., 1:])

            for res, feature_vec in zip(results, features_norm):
                corpus_index = id_to_index[res[0]]
                scores[corpus_index] = np.dot(feature_vec, norm_weights)

    return scores

def get_word_weights(corpus_under_topic, dictionary, topic_id, model, 
                 normalize, scores, alpha=0.7):
    '''
    Computes word importance in a weighted corpus
    Args:
    corpus_under_topic: Corpus_under_topic object for a given topic 
    topic_id:           integer topic identifier
    model:              language model to convert corpus to a desirable 
                        represention (e.g., tf-idf)
    normalize:          whether to normalize the representation 
    scores:             list of reply scores under each topic
    alpha:              weight associated with topic content itself
    Returns:
    dict of word importance values
    '''
    word_weights = collections.defaultdict(float)

    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_under_topic]
    language_model = model(corpus_bow, normalize=normalize)    

    # get the max score under each topic for normalization purposes
    max_score = max(scores)
    for i, doc in enumerate(corpus_bow):
        converted = language_model[doc]
        max_word_weight = max([x[1] for x in converted])
        coeff, score_norm = 1-alpha, scores[i]/max_score if i else alpha, 1 
        for word in converted:
            word_weight_norm = word[1]/max_word_weight
            word_weights[word[0]] += coeff*score_norm*word_weight_norm

    return word_weights

def get_word_weights_all_topics(db, tid_to_table, features, weights, 
                                preprocess_fn, stopwords, normalize):

    '''
    Computes word weight dictionary for all discussion threads
    Args:
    db:            database connection
    tid_to_table:  dictioanry mapping topic id's to table numbers
    features:      attributes to include in importance evaluation
    weights:       weights associated with attributes in features
    preprocess_fn: function to preprocess original text 
    stopwords:     set of stopwords
    model:         language model to convert corpus to a desirable 
                   represention (e.g., tf-idf)
    normalize:     whether to normalize the representation 
    '''
    word_weights = {}

    # create a Corpus_under_topic object for each topic
    for topic_id in tid_to_table:
        corpus = stream.Corpus_under_topic(db, topic_id, 
                                           tid_to_table[topic_id], 
                                           stopwords, preprocess_fn)
        dictionary = corpora.Dictionary(corpus)
        scores = get_scores(db, topic_id, features, weights, 
                            corpus.reply_id_to_corpus_index)        
        word_weights[topic_id] = get_word_weights(
                                 corpus, dictionary, topic_id, 
                                 model, normalize, scores)

    return word_weights 