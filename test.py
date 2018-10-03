import numpy as np
from sklearn import preprocessing

class Stream(object):
    def __init__(self, preprocess_fn, dictionary, stopwords):
        self.preprocess_fn = preprocess_fn
        self.dictionary = dictionary
        self.stopwords = stopwords

    def __iter__(self):
        with open('./sample_corpus.txt', 'r') as f:
            while True:
                text = f.readline().strip()
                if text == '':
                    break
                yield self.dictionary.doc2bow(
                    self.preprocess_fn(text, self.stopwords))

'''
stream = Stream(comment_scoring.preprocess)
dictionary = build_dictionary(stream)

for word_id, word in dictionary.items():
    print(word_id, word)

corpus = [dictionary.doc2bow(text) for text in stream]


for vec in corpus:
    print(vec)
'''
norm_weights = [1,1,1,1]
scaler = preprocessing.MinMaxScaler(copy=False)
replies = [(2,1,5,3,9),
        (21,4,31,19,15),
        (5,2,18,4,8),
        (23,7,4,55,32)]

# normalize features using min-max scaler
features_norm = scaler.fit_transform(np.array(replies)[..., 1:])

print(features_norm)

    
for (reply_id, _, _, _), feature_vec in zip(replies, features_norm):
    print(reply_id, end=' ')
    print(np.dot(feature_vec, norm_weights))





