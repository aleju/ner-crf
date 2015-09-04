# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

def split_to_chunks(of, chunk_size):
    assert of is not None

    for i in range(0, len(of), chunk_size):
        yield of[i:i + chunk_size]

def load_articles(filepath, start_at=0):
    skipped = 0
    with open(filepath, "r") as f:
        for article in f:
            article = article.decode("utf-8").strip()
            
            if len(article) > 0:
                if skipped < start_at:
                    skipped += 1
                else:
                    yield Article(article)

def load_windows(articles, window_size, features=None, every_nth=1, only_labeled_chunks=False):
    processedArticlesSoFar = 0
    nth = 0

    for article in articles:
        counts = article.get_label_counts()
        counts_sum = sum([count[1] for count in counts])
        
        if counts_sum / len(article.tokens) >= 0.10:
            pass
        elif only_labeled_chunks and counts_sum == 0:
            pass
        else:
            token_chunks = split_to_chunks(tokens, chunk_size)
            for token_chunk in token_chunks:
                counts_chunk = count_tags(" ".join(token_chunk))
                if not prefer_labeled_chunks or counts_chunk.sum > 0:
                    window = Window([Token(token) for token in token_chunk])
                    if features is not None:
                        window.apply_features(features)
                    yield window
                nth += 1

"""
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
"""

class Article(object):
    def __init__(self, text):
        text = re.sub(r"[\t ]+", " ", text)
        tokens_str = [token_str.strip() for token_str in text.strip().split(" ")]
        self.tokens = [Token(token_str) for token_str in tokens_str if len(token_str) > 0]
    
    def get_content_as_string():
        return " ".join([token.word for token in self.tokens])
    
    def get_label_counts(add_no_ne_label=False):
        counts = defaultdict(0)
        for token in tokens:
            if token.label != NO_NE_LABEL or add_no_ne_label:
                counts[token.label] += 1
        return counts.iteritems()

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
        self.features_values = None

class Window(object):
    def __init__(self, tokens):
        self.tokens = tokens

    def apply_features(features):
        # returns a multi-dimensional list
        # 1st dimension: Feature (class)
        # 2nd dimension: token
        # 3rd dimension: values (for this token and feature, usually just one value, sometimes more,
        #                        e.g. "w2vc=975")
        features_values = [feature.convert_sentence(self) for feature in features]
        
        for token in self.tokens:
            token.features_values = []
        
        for feature_values in features_values:
            assert type(feature_values) == type(list())
            assert len(feature_values) == len(self.tokens)
            
            for token_idx in range(len(self.tokens)):
                self.tokens[token_idx].features_values.extend(feature_values[token_idx])

"""
def count_tags(article_str):
    counts = {"PER": article_str.count("/PER"),
              "LOC": article_str.count("/LOC"),
              "ORG": article_str.count("/ORG"),
              "MISC": article_str.count("/MISC")}
    counts["max"] = max([count for count in counts.itervalues()])
    counts["sum"] = sum([count for count in counts.itervalues()])
    return counts
"""
