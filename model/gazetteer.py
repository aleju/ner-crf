# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

class Gazetteer():
    def __init__(self, unigrams_names, unigrams):
        self.gazetteer = set()
        self.fill_by_comparison(unigrams_names, unigrams)
    
    def clear(self):
        self.gazetteer = set()
    
    def fill_by_comparison(self, unigrams_names, unigrams):
        sum_names_counts = unigrams_names.sum_of_counts
        sum_ug_counts = unigrams.sum_of_counts

        for name, _ in unigrams_names.word_to_count.iteritems():
            freq_names = unigrams_names.get_frequency_of(name)
            freq_all = unigrams.get_frequency_of(name)
            
            if freq_all is None or freq_names > freq_all:
                self.gazetteer.add(name)
    
    def contains(self, word):
        return (word in self.gazetteer)
