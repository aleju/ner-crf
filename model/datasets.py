# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

def load_articles(start_at=0, filepath):
    skipped = 0
    with open(filepath, 'rU') as f:
        for article in f:
            article = article.decode("utf-8").strip()
            
            if len(article) > 0:
                if skipped < start_at:
                    skipped += 1
                else:
                    yield article

def articles_to_xy(articles_gen, chunk_size, every_nth=1, prefer_labeled_chunks=False, active_features=None):
    processedArticlesSoFar = 0
    nth = 0

    for article in articles_gen:
        tokens = article.split(" ")
        counts = count_tags(article)
        
        if counts / len(tokens) >= 0.10:
            pass
        elif prefer_labeled_chunks and counts.sum == 0:
            pass
        else:
            token_chunks = to_chunks(tokens, chunk_size)
            for token_chunk in token_chunks:
                counts_chunk = count_tags(" ".join(token_chunk))
                if not prefer_labeled_chunks or counts_chunk.sum > 0:
                    sentence = Sentence([Token(token) for token in token_chunk])
                    #(features, labels, tokens_orig, tokens_unicode, tokens_ascii) = text_to_xy(token_chunk, active_features=active_features)
                    yield (features, labels, tokens_unicode)
                nth += 1

def tokens_to_xy(tokens, window_left, window_right, active_features=None):
    tokensOrig = []
    tokensInUnicode = []
    tokens = []
    features = []
    labels = []

    tokens_orig = tokens
    tokens_cleaned = list()
    tokens_unicode = list(tokens)

    labels = [NO_NE_LABEL] * len(tokens)

    for i, token in enumerate(tokens_orig):
        pos = token.rfind("/")
        if pos > -1:
            posUnicode = tokensInUnicode[i].rfind("/")
            end = token[-(len(token) - pos) + 1:]
            if end in ["PER"]:
                tokens[i] = token[0:pos]
                tokensInUnicode[i] = tokensInUnicode[i][0:posUnicode]
                labels[i] = end
            elif end in ["LOC", "ORG", "MISC"]:
                tokens[i] = token[0:pos]
                tokensInUnicode[i] = tokensInUnicode[i][0:posUnicode]
                labels[i] = NO_NE_LABEL

    features = tokens2features(tokens, tokensInUnicode, windowLeftSize, windowRightSize, activeFeatures=activeFeatures)

    return (features, labels, tokensInUnicode, tokens)

class Token(object):
    def __init__(self, original):
        self.original = original
        pos = token.rfind("/")
        end = token[pos+1:]
        if pos > -1 and end in ["PER", "LOC", "ORG", "MISC"]:
            self.word = original[0:pos]
            self.label = end
        else:
            self.word = original
            self.label = NO_NE_LABEL
        self.word_ascii = cleanupUnicode(self.word)

class Sentence(object):
    def __init__(self, tokens):
        self.tokens = tokens

    def get_feature_windows(window_left_size, window_right_size, active_features=None):
        

class FeatureWindow(object):
    def __init__(self, tokens, features):
        

def count_tags(article_str):
    counts = {"PER": article_str.count("/PER"),
              "LOC": article_str.count("/LOC"),
              "ORG": article_str.count("/ORG"),
              "MISC": article_str.count("/MISC")}
    counts["max"] = max([count for count in counts.itervalues()])
    counts["sum"] = sum([count for count in counts.itervalues()])
    return counts
