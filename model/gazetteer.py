# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

class Gazetteer():
    def __init__(self, unigrams_names_filepath, unigrams_filepath):
        self.gazetteer = dict()
        self.fill_by_comparison(unigrams_names_filepath, unigrams_filepath)
    
    def clear(self):
        self.gazetteer = dict()
    
    def fill_by_comparison(self, unigrams_names_filepath, unigrams_filepath):
        unigrams = UnigramDict(unigrams_filepath)
        unigrams_names = UnigramList(unigrams_names_filepath)
        
        sum_ug_counts = 0
        for count in unigrams.word_to_count.itervalues():
            sum_ug_counts += count
        
        sum_names_counts = 0
        for count in unigrams_names.word_to_count.itervalues():
            sum_names_counts += count

        for name, count in unigrams_names.word_to_count.iteritems():
            count_in_unigrams = unigrams.get_count_of(name, 0)
            if count_in_unigrams == 0:
                # if the name does appear among the list of names but not among all unigrams,
                # we add it right away (this scenario usually shouldn't happen)
                self.gazetteer.add(name)
            else:
                # how common is the name among all names
                frac_names = count / sum_names_counts
                # how common is the name among all words
                frac_unigrams = count_in_unigrams / sum_ug_counts
                
                # the name is more common among all names than among the normal corpus
                # so we add it to the list of names
                if frac_names > frac_unigrams:
                    self.gazetteer.add(name)
    
    def is_in_gazetteer(self, word):
        return (word in self.gazetteer)
