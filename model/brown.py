# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

class BrownClusterDict():
    def __init__(self, filepath):
        self.word_to_cluster = dict()
        self.word_to_bitchain = dict()
        self.fill_from_file(filepath)
    
    def clear(self):
        self.word_to_cluster = dict()
        self.word_to_bitchain = dict()
    
    def fill_from_file(self, filepath):
        with open(filepath, "r") as f:
            last_count = -1
            cluster_idx = 1
            
            for line in f:
                columns = line.decode("utf-8").strip().split("\t")
                if len(columns) == 3:
                    bitchain, word, count = columns
                    count = int(count)
                    
                    if count < last_count:
                        cluster_idx += 1
                    last_count = count
                    
                    self.word_to_cluster[word] = cluster_idx
                    self.word_to_bitchain[word] = bitchain

    def get_cluster_of(self, word, default=-1):
        if word in self.word_to_cluster:
            return self.word_to_cluster[word]
        else:
            return default

    def get_bitchain_of(self, word, default=-1):
        if word in self.word_to_bitchain:
            return self.word_to_bitchain[word]
        else:
            return default
