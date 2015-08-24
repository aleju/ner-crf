# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

class UnigramDict():
    def __init__(self, filepath, skip_first_n=0, max_count_words=None):
        self.word_to_rank = dict()
        self.word_to_count = dict()
        self.fill_from_file(filepath)
    
    def clear(self):
        self.word_to_rank = dict()
        self.word_to_count = dict()
    
    def fill_from_file(self, filepath, skip_first_n=0, max_count_words=None):
        skipped = 0
        added = 0
        with open(filepath, "r") as f:
            for line in f:
                columns = line.decode("utf-8").strip().split("\t")
                if len(columns) == 2:
                    if skipped < skip_first_n:
                        skipped += 1
                    else:
                        if added is not None and added >= max_count_words:
                            break
                        else:
                            word, count = columns
                            count = int(count)
                            self.word_to_rank[word] = added + 1
                            self.word_to_count[word] = count
                            added += 1
    
    def get_rank_of(self, word, default=-1):
        if word in self.word_to_rank:
            return self.word_to_rank[word]
        else:
            return default
    
    def get_count_of(self, word, default=-1):
        if word in self.word_to_count:
            return self.word_to_count[word]
        else:
            return default
    
