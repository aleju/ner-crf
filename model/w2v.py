# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

class W2VClusters():
    def __init__(self, filepath):
        self.word_to_cluster = dict()
        self.fill_from_file(filepath)
    
    def clear(self):
        self.word_to_cluster = dict()
    
    def fill_from_file(self, filepath):
        with open(filepath, "r") as f:
            for line in f:
                columns = line.decode("utf-8").strip().split("\t")
                if len(columns) == 2:
                    word, cluster_idx = columns
                    self.word_to_cluster[word] = cluster_idx
    
    def get_cluster_of(self, word, default=-1):
        if word in self.word_to_cluster:
            return self.word_to_cluster[word]
        else:
            return default
