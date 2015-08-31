# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import Counter
from datasets import load_articles

class Unigrams():
    def __init__(self, filepath=None, skip_first_n=0, max_count_words=None):
        self.word_to_rank = dict()
        self.word_to_count = dict()
        self.sum_of_counts = 0
        if filepath is not None:
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
                            self.sum_of_counts += count
                            added += 1
    
    def fill_names_from_file(self, filepath, labels):
        assert type(labels) == type(list())
        assert len(labels) > 0
        
        counts = Counter()
        self.sum_of_counts = 0
        
        gen = load_articles(filepath, start_at=0)
        for article in gen:
            for token in article:
                if token.label in labels:
                    counts[token.word] += 1
        
        most_common = counts.most_common()
        self.word_to_count = dict(most_common)
        for i, (name, count) in enumerate(most_common):
            self.word_to_rank[name] = i + 1
            self.sum_of_counts += count
    
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
    
    def get_frequency_of(self, word, default=None):
        count = self.get_count_of(word, None)
        if count is not None:
            return count / self.sum_of_counts
        else:
            return default
