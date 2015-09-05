# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import Counter, OrderedDict
from model.datasets import load_articles

class Unigrams():
    def __init__(self, filepath=None, skip_first_n=0, max_count_words=None):
        self.word_to_rank = OrderedDict()
        self.word_to_count = OrderedDict()
        self.sum_of_counts = 0
        if filepath is not None:
            self.fill_from_file(filepath, skip_first_n=skip_first_n, max_count_words=max_count_words)
    
    def clear(self):
        self.word_to_rank = OrderedDict()
        self.word_to_count = OrderedDict()
    
    def fill_from_file(self, filepath, skip_first_n=0, max_count_words=None):
        skipped = 0
        added = 0
        with open(filepath, "r") as f:
            for line_idx, line in enumerate(f):
                columns = line.decode("utf-8").strip().split("\t")
                if len(columns) == 2:
                    if skipped < skip_first_n:
                        skipped += 1
                    else:
                        if max_count_words is not None and added >= max_count_words:
                            break
                        else:
                            word, count = columns
                            count = int(count)
                            self.word_to_rank[word] = added + 1
                            self.word_to_count[word] = count
                            self.sum_of_counts += count
                            added += 1
                else:
                    print("[Warning] Expected 2 columns in unigrams file at line %d, got %d" % (line_idx, len(columns)))
    
    def fill_from_articles(self, filepath, verbose=False):
        self.fill_from_articles_labels(filepath, labels=None, verbose=verbose)
    
    def fill_from_articles_labels(self, filepath, labels=None, verbose=False):
        assert labels is None or type(labels) == type(list())
        assert labels is None or len(labels) > 0
        
        counts = Counter()
        self.sum_of_counts = 0
        
        articles = load_articles(filepath, start_at=0)
        for i, article in enumerate(articles):
            words = [token.word for token in article.tokens if labels is None or token.label in labels]
            counts.update(words)
            if verbose and i % 1000 == 0:
                print("Article %d" % (i))
        
        most_common = counts.most_common()
        for i, (word, count) in enumerate(most_common):
            self.word_to_count[word] = count
            self.word_to_rank[word] = i + 1
            self.sum_of_counts += count
    
    def write_to_file(self, filepath):
        with open(filepath, "w") as f:
            for i, (word, count) in enumerate(self.word_to_count.iteritems()):
                print(word, count)
                if i > 0:
                    f.write("\n")
                f.write(word.encode("utf-8"))
                f.write("\t")
                f.write(str(count))
    
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
