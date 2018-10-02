import test
import utilities
from gensim import corpora

STOPWORDS = './stopwords.txt'
TEXTS = ['工作压力很大',
         'NBA季后赛快开始了',
         '王者荣耀',
         '杜兰特太厉害了'
         ]

def main():
    texts = [utilities.preprocess(text, STOPWORDS) for text in TEXTS]
    dictionary = {1:1, 2:2, 3:3}
    
    for id_, word in dictionary.items():
        print(id_, word)
    
    stream = test.Stream(utilities.preprocess, dictionary, STOPWORDS)
    for s in stream:
        print(s)

if __name__ == '__main__':
    main()