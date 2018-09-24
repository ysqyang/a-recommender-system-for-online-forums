import networkx as nx
import sklearn
import jieba
from gensim import corpora, models, similarities

def tokenize(sentence, stopwords_dict_path):
    '''
    Tokenize a Chinese sentence to a list of words
    Args:
    sentence: sentence to be tokenized
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

    print(stopwords)

    return [word for word in jieba.lcut(sentence) if word not in stopwords]  

def compute_similarity(sentence_1, sentence_2):
    '''
    Computes the similarity between two sentences
    Args:
    sentence_1: a Chinese sentence tokenized into words
    sentence_2: a Chinese sentence tokenized into words
    '''
    



def make_similarity_graph(sentences, similarity_fn):
    '''
    Creates a undirected graph with sentences as nodes.
    An edge exists between two sentences if and only they
    have a nonzero similarity score.
    Args:
    sentences: list of sentences for which pairwise similarities
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
    G: graph for which the PageRank is to be computed
    alpha: damping coefficient in the PageRank algorithm
    '''
    return nx.pagerank(G, alpha=alpha)

def compute_comment_importance(graphs, coefficients, alphas):
    '''
    Computes comment importance by linearly combining
    the PageRank for graphs in G 
    Args:
    graphs: list of graphs
    coefficients: coefficients of the linear combination
    alphas: damping factors used in the PageRank algorithm
    '''
    return sum([c*pagerank(g, a) for g, c, a in 
               zip(graphs, coefficients, alphas)])

















