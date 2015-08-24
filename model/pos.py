# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import nltk

class PosTagger():
    def __init__(stanford_postagger_jar_filepath, stanford_model_filepath, cache_dir=None):
        self.cache_synch_prob = 2 # in percent, 1 to 100
        self.max_string_length = 2000
        self.min_string_length = 1
        
        self.tagger = nltk.tag.stanford.POSTagger(stanford_postagger_jar_filepath,
                                                  stanford_model_filepath,
                                                  encoding="utf-8")
        self.cache_dir = cache_dir
        self.cache = shelve.open(cache_dir) if cache_dir is not None else None
    
    def tag(text):
        if self.cache is None:
            return self.tag_uncached(text)
        else:
            _hash = str(hash(text))
            if self.cache.has_key(_hash):
                return self.cache[_hash]
            else:
                tagged = self.tag_uncached(text)
                
                self.cache[_hash] = tagged
                if random.randint(1, 100) <= self.cache_synch_prob:
                    self.synchronize_cache()
            
                return tagged
    
    
    def tag_uncached(text):
        if len(text) >= self.max_string_length:
            raise Exception("String to POS-tag is too long (%d vs max " \
                            "%d)." % (len(text), self.max_string_length))
        elif len(text) < self.min_string_length:
            raise Exception("String to POS-tag is too short (%d vs min "\
                            "%d)." % (len(text), self.min_string_length))
        
        return self.tagger.tag(text.split())
    
    def synchronize_cache():
        self.cache.synch()
