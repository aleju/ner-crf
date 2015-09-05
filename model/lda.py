# -*- coding: utf-8 -*-
"""Class that wraps a previously trained gensim LDA."""
from __future__ import absolute_import, division, print_function, unicode_literals
import random
import shelve
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore

class LdaWrapper():
    """Class that wraps a previously trained gensim LDA.
    
    This class uses a shelve cache to store generated results. This speeds up the generation
    of training examples, if the identical corpus, window sizes etc. are used.
    """
    def __init__(self, lda_filepath, dictionary_filepath, cache_filepath=None):
        """Initialize the LDA wrapper.
        Args:
            lda_filepath: Filepath to the trained LDA model.
            dictionary_filepath: Filepath to the dictionary of the LDA.
            cache_filepath: Optional filepath to a shelve cache for the LDA results.
        """
        self.lda = LdaMulticore.load(lda_filepath)
        self.dictionary = gensim.corpora.dictionary.Dictionary.load(dictionary_filepath)
        self.cache_synch_prob = 2 # in percent, 1 to 100
        self.cache_filepath = cache_filepath
        self.cache = shelve.open(cache_filepath) if cache_filepath is not None else None
    
    def get_topics(self, text):
        """Returns the topics of a small string text window.
        Args:
            text: A small text window as string.
        Returns:
            List of tuples of form (topic index, probability).
        """
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
        """Returns the topics of a small string text window without querying the cache.
        Args:
            text: A small text window as string.
        Returns:
            List of tuples of form (topic index, probability).
        """
        tokens = text.lower().split(" ")
        return self.lda[self.dictionary.doc2bow(tokens)]

    def synchronize_cache(self):
        """Synchronizes the shelve cache on the HDD with the version in the RAM."""
        self.cache.sync()
