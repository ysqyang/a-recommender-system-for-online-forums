import networkx as nx
import sklearn
import jieba
from gensim import corpora, models, similarities
import numpy as np

def tokenize(sentence, stopwords_dict_path):
    '''
    Tokenize a Chinese sentence to a list of words
    Args:
    sentence:       sentence to be tokenized
    stopwords_dict: path for the stopword dictionary
    '''
    stopwords = set()
    # load stopwords dictionary
    with open(stopwords_dict_path, 'r') as f:
        while True:
            stopword = f.readline().strip()
            if stopword == '':
                break
            stopwords.add(stopword)

    return [word for word in jieba.lcut(sentence) if word not in stopwords]  

def sentence_vector(sentence, tokenize_fn):
    '''
    Converts a Chinese sentence to a vector using word embeddings
    Args:
    sentence:    sentence to be converted
    tokenize_fn: function to tokenize a sentence
    '''
    words = tokenize_fn(sentence)
    v = np.zeros(64)      
    
    for word in words:
        v += model[word]
    
    v /= len(words)
    
    return v

def sentence_similarity(sentence_1, sentence_2, sentence_to_vector_fn, word2vec_model_path):
    '''
    Computes the similarity between two sentences
    Args:
    sentence_1:             a Chinese sentence tokenized into words
    sentence_2:             a Chinese sentence tokenized into words
    sentence_to_vector_fn:  function to convert a sentence to a vector 
    word2vec_model_path:    path of binary word2vec model file
    '''
    model = models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)    
    v1, v2 = sentence_to_vector_fn(sentence1), sentence_to_vector_fn(sentence2)

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def make_similarity_graph(sentences, similarity_fn):
    '''
    Creates a undirected graph with sentences as nodes.
    An edge exists between two sentences if and only they
    have a nonzero similarity score.
    Args:
    sentences:     list of sentences for which pairwise similarities
                   are to be computed
    similarity_fn: function to compute the similarity between two sentences
    '''

    G = nx.Graph()  # similarity is symmetric, so the graph is undirected
    G.add_nodes_from(sentences)
    num_sentences = len(sentences)
    for i in range(num_sentences):
        for j in range(i+1, num_sentences):
            G.add_edge(sentences[i], sentences[j], 
                       weight=similarity_fn(sentences[i], sentences[j]))

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
          '今天天气好热',
          '他们在犹豫晚上要不要出去吃饭',
          '你后天要去北京',
          '她儿子考上了清华大学',
          '最近股票跌得很厉害',
          '今年高考她没考好',
          '这两年经济不景气啊',
          '下周要下雪了']

corpus = [tokenize(doc, './stopwords.txt') for doc in corpus]

dictionary = corpora.Dictionary(corpus)

corpus = [dictionary.doc2bow(text) for text in corpus]
print(corpus)

tfidf = models.TfidfModel(corpus)

print(tfidf[corpus[0]])















