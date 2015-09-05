# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import random
import shelve
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore

class LdaWrapper():
    def __init__(self, lda_filepath, dictionary_filepath, cache_filepath=None):
        self.lda = LdaMulticore.load(lda_filepath)
        self.dictionary = gensim.corpora.dictionary.Dictionary.load(dictionary_filepath)
        #self.cache = dict()
        #self.cache_max_size = cache_max_size
        self.cache_synch_prob = 2 # in percent, 1 to 100
        self.cache_filepath = cache_filepath
        self.cache = shelve.open(cache_filepath) if cache_filepath is not None else None
    
    def get_topics(self, text):
        if self.cache is None:
            return self.get_topics_uncached(text)
        else:
            _hash = str(hash(text))

            if self.cache.has_key(_hash):
                return self.cache[_hash]
            else:
                topics = self.get_topics_uncached(text)
                
                self.cache[_hash] = topics
                if random.randint(1, 100) <= self.cache_synch_prob:
                    self.synchronize_cache()
                
                return topics
            
    def get_topics_uncached(self, text):
        tokens = text.lower().split(" ")
        return self.lda[self.dictionary.doc2bow(tokens)]

    def synchronize_cache(self):
        self.cache.sync()
