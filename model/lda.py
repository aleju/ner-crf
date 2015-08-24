# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore

class LdaWrapper():
    def __init__(lda_filepath, dictionary_filepath, cache_max_size=0):
        self.lda = LdaMulticore.load(lda_filepath)
        self.dictionary = gensim.corpora.dictionary.Dictionary.load(dictionary_filepath)
        self.cache = dict()
        self.cache_max_size = cache_max_size
    
    def get_topics(self, text):
        if self.cache_max_size <= 0:
            return self.get_topics_uncached(text)
        else:
            _hash = hash(text)

            if _hash in self.cache:
                return self.cache[_hash]
            else:
                topics = self.get_topics_uncached(text)
                
                if len(self.cache) > self.cache_max_size:
                    self.cache.pop(random.choice(self.cache.keys()))
                self.cache[_hash] = topics
                
                return topics
            
    def get_topics_uncached(self, text):
        tokens = text.lower().split(" ")
        return self.lda[self.dictionary.doc2bow(tokens)]
