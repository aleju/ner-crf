# -*- coding: utf-8 -*-
"""Functions to load data from the corpus."""
from __future__ import absolute_import, division, print_function, unicode_literals
import re
from unidecode import unidecode
from collections import Counter
from config import *

def split_to_chunks(of, chunk_size):
    """Splits a list to smaller chunks.
    Args:
        of: The list/iterable to split.
        chunk_size: The maximum size of each smaller part/chunk.
    Returns:
        Generator of lists (i.e. list of lists).
    """
    assert of is not None

    for i in range(0, len(of), chunk_size):
        yield of[i:i + chunk_size]

def load_articles(filepath, start_at=0):
    """Loads all articles (documents) from a corpus.
    
    The corpus is expected to be a UTF-8 encoded textfile with one article/document per line.
    
    Args:
        filepath: The filepath to the corpus file.
        start_at: The index of the line to start at. (Default is 0.)
    Returns:
        Generator of Article objects, i.e. list of Article.
    """
    skipped = 0
    with open(filepath, "r") as f:
        for article in f:
            article = article.decode("utf-8").strip()
            
            if len(article) > 0:
                if skipped < start_at:
                    skipped += 1
                else:
                    yield Article(article)

def load_windows(articles, window_size, features=None, every_nth_window=1, only_labeled_windows=False):
    """Loads smaller windows with a maximum size per window from a generator of articles.
    
    Args:
        articles: Generator of articles, as provided by load_articles().
        window_size: Maximum length of each window (in tokens/words).
        features: Optional features to apply to each window. (Default is None, don't apply
            any features.)
        every_nth_window: How often windows are ought to be returned, e.g. a value of 3 will
            skip two windows and return the third one. This can spread the examples over more
            (different) articles. (Default is 1, return every window.)
        only_labeled_windows: If set to True, the function will only return windows that contain
            at least one labeled token (at leas one named entity). (Default is False.)
    Returns:
        Generator of Window objects, i.e. list of Window objects.
    """
    processed_windows = 0
    for article in articles:
        count = article.count_labels()
        #counts = article.get_label_counts()
        #counts_sum = sum([count[1] for count in counts])
        
        if count / len(article.tokens) >= 0.10:
            pass
        elif only_labeled_windows and count == 0:
            pass
        else:
            token_windows = split_to_chunks(article.tokens, window_size)
            for token_window in token_windows:
                window = Window([token for token in token_window])
                if not only_labeled_windows or window.count_labels() > 0:
                    if processed_windows % every_nth_window == 0:
                        if features is not None:
                            window.apply_features(features)
                        yield window
                    processed_windows += 1

def generate_examples(windows, nb_append=None, nb_skip=0, verbose=True):
    """Generates example pairs of feature lists (one per token) and labels.
    
    Args:
        windows: The windows to generate features and labels from, see load_windows().
        nb_append: How many windows to append max or None if unlimited. (Default is None.)
        nb_skip: How many windows to skip at the start. (Default is 0.)
        verbose: Whether to print status messages. (Default is True.)
    Returns:
        Pairs of (features, labels),
        where features is a list of lists of strings,
            e.g. [["foo=bar", "asd=fgh"], ["foo=not_bar", "yikes=True"], ...]
        and labels is a list of strings,
            e.g. ["PER", "O", "O", "LOC", ...].
    """
    skipped = 0
    added = 0
    for window in windows:
        # skip the first couple of windows, if nb_skip is > 0
        if skipped < nb_skip:
            skipped += 1
        else:
            # chain of labels (list of strings)
            labels = window.get_labels()
            # chain of features (list of lists of strings)
            feature_values_lists = []
            for word_idx in range(len(window.tokens)):
                fvl = window.get_feature_values_list(word_idx, SKIPCHAIN_LEFT, SKIPCHAIN_RIGHT)
                feature_values_lists.append(fvl)
            # yield (features, labels) pair
            yield (feature_values_lists, labels)
            
            # print message every nth window
            # and stop if nb_append is reached
            added += 1
            if verbose and added % 500 == 0:
                if nb_append is None:
                    print("Generated %d examples" % (added, nb_append))
                else:
                    print("Generated %d of max %d examples" % (added, nb_append))
            if nb_append is not None and added == nb_append:
                break

def cleanup_unicode(in_str):
    """Converts unicode strings to ascii.
    The function uses mostly unidecode() and contains some additional mappings for german umlauts.
    
    Args:
        in_str: String in UTF-8.
    Returns:
        String (ascii).
    """
    result = in_str

    mappings = [(u"ü", "ue"), (u"ö", "oe"), (u"ä", "ae"),
                (u"Ü", "Ue"), (u"Ö", "Oe"), (u"Ä", "Ae"),
                (u"ß", "ss")]
    for str_from, str_to in mappings:
        result = result.replace(str_from, str_to)

    result = unidecode(result)

    return result

class Article(object):
    """Class modelling an article/document from the corpus. It's mostly a wrapper around a list
    of Token objects."""
    def __init__(self, text):
        """Initialize a new Article object.
        Args:
            text: The string content of the article/document.
        """
        text = re.sub(r"[\t ]+", " ", text)
        tokens_str = [token_str.strip() for token_str in text.strip().split(" ")]
        self.tokens = [Token(token_str) for token_str in tokens_str if len(token_str) > 0]
    
    def get_content_as_string(self):
        """Returns the article's content as string.
        This is not neccessarily identical to the original text content, because multi-whitespaces
        are replaced by single whitespaces.
        
        Returns:
            string (article/document content).
        """
        return " ".join([token.word for token in self.tokens])
    
    def get_label_counts(self, add_no_ne_label=False):
        """Returns the count of each label in the article/document.
        Count means here: the number of words that have the label.
        
        Args:
            add_no_ne_label: Whether to count how often unlabeled words appear. (Default is False.)
        Returns:
            List of tuples of the form (label as string, count as integer).
        """
        if add_no_ne_label:
            counts = Counter([token.label for token in self.tokens])
        else:
            counts = Counter([token.label for token in self.tokens if token.label != NO_NE_LABEL])
        return counts.most_common()
    
    def count_labels(self, add_no_ne_label=False):
        """Returns how many named entity tokens appear in the article/document.
        
        Args:
            add_no_ne_label: Whether to also count unlabeled words. (Default is False.)
        Returns:
            Count of all named entity tokens (integer).
        """
        return sum([count[1] for count in self.get_label_counts(add_no_ne_label=add_no_ne_label)])

class Window(Article):
    """Encapsulates a small window of text/tokens."""
    def __init__(self, tokens):
        """Initialize a new Window object.
        
        Args:
            tokens: The tokens/words contained in the text window, provided as list of Token
                objects.
        """
        self.tokens = tokens

    def apply_features(self, features):
        """Applies a list of feature generators to the tokens of this window.
        Each feature generator will then generate a list of featue values (as strings) for each
        token. Each of these lists can be empty. The lists are saved in the tokens and can later
        on be requested multiple times without the generation overhead (which can be heavy for
        some features).
        
        Args:
            features: A list of feature generators from features.py .
        """
        # feature_values is a multi-dimensional list
        # 1st dimension: Feature (class)
        # 2nd dimension: token
        # 3rd dimension: values (for this token and feature, usually just one value, sometimes more,
        #                        e.g. "w2vc=975")
        features_values = [feature.convert_window(self) for feature in features]
        
        for token in self.tokens:
            token.feature_values = []
        
        # After this, each self.token.feature_values will be a simple list
        # of feature values, e.g. ["w2v=875", "bc=48", ...]
        for feature_values in features_values:
            assert type(feature_values) == type(list())
            assert len(feature_values) == len(self.tokens)
            
            for token_idx in range(len(self.tokens)):
                self.tokens[token_idx].feature_values.extend(feature_values[token_idx])
    
    def get_feature_values_list(self, word_index, skipchain_left, skipchain_right):
        """Generates a list of feature values (strings) for one token/word in the window.
        
        Args:
            word_index: The index of the word/token for which to generate the featueres.
            skipchain_left: How many words to the left will be included among the features of
                the requested word. E.g. a value of 1 could lead to a list like
                ["-1:w2vc=123", "-1:l=30", "0:w2vc=18", "0:l=4"].
            skipchain_right: Like skipchain_left, but to the right side.
        Returns:
            List of strings (list of feature values).
        """
        assert word_index >= 0
        assert word_index < len(self.tokens)

        all_feature_values = []

        start = max(0, word_index - skipchain_left)
        end = min(len(self.tokens), word_index + 1 + skipchain_right)
        for i, token in enumerate(self.tokens[start:end]):
            diff = start + i - word_index
            feature_values = ["%d:%s" % (diff, feature_value) for feature_value in token.feature_values]
            all_feature_values.extend(feature_values)

        return all_feature_values

    def get_labels(self):
        return [token.label for token in self.tokens]

class Token(object):
    """Encapsulates a token/word.
    Members:
        token.original: The original token, i.e. the word _and_ the label, e.g. "John/PER".
        token.word: The string content of the token, without the label.
        token.label: The label of the token.
        token.feature_values: The feature values, after they have been applied.
            (See Window.apply_features().)
    """
    def __init__(self, original):
        """Initialize a new Token object.
        Args:
            original: The original word as found in the text document, including the label,
                e.g. "foo", "John/PER".
        """
        self.original = original
        self.word = original
        self.label = NO_NE_LABEL
        if "/" in original:
            pos = original.rfind("/")
            end = original[pos+1:]
            # remove parts of BIO encoding, e.g. remove "B-" from "B-PER" or "I-" from "I-PER"
            if REMOVE_BIO_ENCODING:
                end = end.replace("B-", "").replace("I-", "")
            if end in LABELS:
                self.word = original[0:pos]
                self.label = end
        self._word_ascii = None
        self.feature_values = None
    
    @property
    def word_ascii(self):
        """Get the ascii value of the token.
        This has its own function as it takes time to compute and isn't even used currently.
        Returns:
            Token string as ASCII.
        """
        if self._word_ascii is None:
            self._word_ascii = cleanup_unicode(self.word)
        return self._word_ascii
