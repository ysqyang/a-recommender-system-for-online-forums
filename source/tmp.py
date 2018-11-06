    def get_distributions(self, coeff):
        '''
        Computes the word distribution for each document in the collection 
        from in-document and in-corpus frequencies
        Args:
        coeff: contribution coefficient for in-document word frequency
               in computing word distribution
        '''
        n_words = len(self.dictionary.token2id)
        self.distributions = []
        corpus_freq = [0]*n_words
        num_tokens_corpus = sum(sum(x[1] for x in vec) for vec in self.corpus_bow)

        for i, vec in enumerate(self.corpus_bow):
            # ignore documents that have less than _MIN_WORDS words 
            if len(vec) == 0:
                self.distributions.append([])
            else:
                # total number of tokens (with repetitions) in current doc
                num_tokens = sum(x[1] for x in vec)
                dst = [0]*n_words
                for (word_id, count) in vec:
                    dst[word_id] = coeff*count/num_tokens
                    # update word frequency in corpus
                    corpus_freq[word_id] += count/num_tokens_corpus
                self.distributions.append(dst)
                '''
                dst = [(self.dictionary[i], val) for i, val in enumerate(dst)]
                dst.sort(key=lambda x:x[1], reverse=True)
                print([(word, val) for word, val in dst if val > 0])
                '''
        for i, dst in enumerate(self.distributions):
            if len(dst) > 0:
                for word_id in range(n_words):
                    # add contribution from in-corpus frequency
                    dst[word_id] += (1-coeff)*corpus_freq[word_id]
                '''
                dst1 = [(self.dictionary[i], val) for i, val in enumerate(dst)]
                dst1.sort(key=lambda x:x[1], reverse=True)
                print([word for word, val in dst1[:20]])
                '''

    def get_distribution_given_profile(self, profile_words):
        '''
        Computes the word distribution given the apprearance of profile_words
        Args:
        profile_words: list of words that represent a discussion thread
                       (i.e., topic with all replies)
        Returns:
        Word distribution given the apprearance of profile_words
        '''
        profile_wids = []
        for word in profile_words:
            if word in self.dictionary.token2id:
                profile_wids.append(self.dictionary.token2id[word])

        #print(profile_word_ids)
        distribution = [0]*len(self.dictionary.token2id)
        # compute the joint probability of observing each dictionary
        # word together with profile words 

        # convert to natural log to avoid numerical issues
        log_probs = [sum(math.log(v[wid]) for wid in profile_wids)
                     if len(v) > 0 else -float('inf') for v in self.distributions]
        # assuming uniform prior distribution over all docs in corpus,
        # the joint probability is the sum of joint probabilities over
        # all docs       
        for v, log_prob in zip(self.distributions, log_probs):
            #print(log_prob)
            for wid in range(len(v)): 
                if wid not in profile_wids:
                    distribution[wid] += math.exp(log_prob+math.log(v[wid]))

        # normalize the probabilities
        s = sum(distribution)
        for i in range(len(distribution)):
            distribution[i] /= s
        
        return distribution

    def get_similarity_given_distribution(self, distribution, T):
        '''
        Computes the similarity scores between a topic profile and 
        the documents in the corpus with time adjustments
        Args:
        distribution: word probability distribution given a topic profile
        T:            time attenuation factor
        Returns:
        Similarity scores between a topic profile and the documents 
        in the corpus
        '''
        sim, now = {}, datetime.now() 
        for date, vec, tid in zip(self.dates, self.distributions, self.valid_topics):
            if len(vec) > 0:     
                date = datetime.strptime(date, self.datetime_format)
                day_diff = (now - date).days
                sim[tid] = stats.entropy(pk=distribution, qk=vec)*math.exp(day_diff/T)
        return sim