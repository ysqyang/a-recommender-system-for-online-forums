import networkx as nx
import sklearn
import jieba
from gensim import corpora, models, similarities
import numpy as np

def get_stopwords(path):
    '''
    Creates stopword set in memory from a file 
    specified by the path arg
    Args:
    path: path for the file containing stopwords
    '''
    n = 0
    stopwords = set()
    # load stopwords dictionary
    with open(path, 'r') as f:
        while True:
            stopword = f.readline().rstrip('\n')
            if stopword == '':
                break
            n += 1
            stopwords.add(stopword)

    return stopwords

def tokenize(text, stopwords_set):
    '''
    Tokenize Chinese text to a list of words
    Args:
    text:          text to be tokenized
    stopwords_set: set of stopwords
    '''
    return [word for word in jieba.lcut(text) if word not in stopwords_set]  

def text_vector(text, tokenize_fn):
    '''
    Converts Chinese text to a vector using word embeddings
    Args:
    text:        text to be converted
    tokenize_fn: function to tokenize a text
    '''
    words = tokenize_fn(text)
    v = np.zeros(64)      
    
    for word in words:
        v += model[word]
    
    return v/len(words)

def text_similarity(text_1, text_2, text_to_vector_fn, word2vec_model_path):
    '''
    Computes the similarity between two texts
    Args:
    text_1:              Chinese text tokenized into words
    text_2:              Chinese text tokenized into words
    text_to_vector_fn:   function to convert a text to a vector 
    word2vec_model_path: path of binary word2vec model file
    '''
    model = models.KeyedVectors.load_word2vec_format(
                                word2vec_model_path, binary=True)

    v1, v2 = text_to_vector_fn(text_1), text_to_vector_fn(text_2)

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def make_similarity_graph(texts, similarity_fn):
    '''
    Creates a undirected graph with texts as nodes.
    An edge exists between two texts if and only they
    have a nonzero similarity score.
    Args:
    texts:         list of text id's for which pairwise similarities
                   are to be computed
    similarity_fn: function to compute the similarity between two texts
    '''

    G = nx.Graph()  # similarity is symmetric, so the graph is undirected
    G.add_nodes_from(texts)
    num_texts = len(texts)
    for i in range(num_texts):
        for j in range(i+1, num_texts):
            G.add_edge(texts[i], texts[j], 
                       weight=similarity_fn(texts[i], texts[j]))

    return G

def pagerank(G, alpha):
    '''
    Computes the PageRank of nodes in graph G
    Args:
    G:     graph for which the PageRank is to be computed
    alpha: damping coefficient in the PageRank algorithm
    '''
    return nx.pagerank(G, alpha=alpha)

def comment_importance(graphs, coefficients, alphas):
    '''
    Computes comment importance by linearly combining
    the PageRank for graphs in G 
    Args:
    graphs:       list of graphs
    coefficients: coefficients of the linear combination
    alphas:       damping factors used in the PageRank algorithm
    '''
    return sum([c*pagerank(g, a) for g, c, a in 
               zip(graphs, coefficients, alphas)])

corpus = ['我昨天去上海了',
          '今天天气好热，不过下周的天气会很舒服',
          '他们在犹豫晚上要不要出去吃饭，后来还是在家里吃饭了',
          '你后天要去北京，北京这几天雾霾很厉害',
          '她儿子考上了清华大学',
          '最近股票跌得很厉害',
          '今年高考她没考好',
          '这两年经济不景气啊',
          '下周要下雪了'
          ]

stopwords = get_stopwords('./stopwords.txt')

corpus = [tokenize(doc, stopwords) for doc in corpus]

dictionary = corpora.Dictionary(corpus)

for id_ in dictionary:
    print(id_, dictionary[id_])

'''
corpus = [dictionary.doc2bow(text) for text in corpus]

print(corpus)
'''
tfidf = models.TfidfModel(corpus)

converted = tfidf[corpus]

for doc in converted:
    print(doc)















