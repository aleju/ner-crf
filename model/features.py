# -*- coding: utf-8 -*-
"""
Contains:
    1. Various classes (feature generators) to convert windows (of words/tokens) to feature values.
       Each feature value is a string, e.g. "starts_with_uppercase=1", "brown_cluster=123".
    2. A method to create all feature generators.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import re

from model.brown import BrownClusters
from model.gazetteer import Gazetteer
from model.lda import LdaWrapper
from model.pos import PosTagger
from model.unigrams import Unigrams
from model.w2v import W2VClusters

# All capitalized constants come from this file
from config import *

def create_features(verbose=True):
    """This method creates all feature generators.
    The feature generators will be used to convert windows of tokens to their string features.
    
    This function may run for a few minutes.
    
    Args:
        verbose: Whether to output messages.
    Returns:
        List of feature generators
    """
    
    def print_if_verbose(msg):
        """This method prints a message only if verbose was set to True, otherwise does nothing.
        Args:
            msg: The message to print.
        """
        if verbose:
            print(msg)
    
    # Load the most common unigrams. These will be used as features.
    print_if_verbose("Loading top N unigrams...")
    ug_all_top = Unigrams(UNIGRAMS_FILEPATH, skip_first_n=UNIGRAMS_SKIP_FIRST_N, max_count_words=UNIGRAMS_MAX_COUNT_WORDS)
    
    # Load all unigrams. These will be used to create the Gazetteer.
    print_if_verbose("Loading all unigrams...")
    ug_all = Unigrams(UNIGRAMS_FILEPATH)
    
    # Load all unigrams of person names (PER). These will be used to create the Gazetteer.
    print_if_verbose("Loading person name unigrams...")
    ug_names = Unigrams(UNIGRAMS_PERSON_FILEPATH)
    
    # Create the gazetteer. The gazetteer will contain all names from ug_names that have a higher
    # frequency among those names than among all unigrams (from ug_all).
    print_if_verbose("Creating gazetteer...")
    gaz = Gazetteer(ug_names, ug_all)
    
    # Unset ug_all and ug_names because we don't need them any more and they need quite a bit of
    # RAM.
    ug_all = None
    ug_names = None
    
    # Load the mapping of word to brown cluster and word to brown cluster bitchain
    print_if_verbose("Loading brown clusters...")
    bc = BrownClusters(BROWN_CLUSTERS_FILEPATH)
    
    # Load the mapping of word to word2vec cluster
    print_if_verbose("Loading W2V clusters...")
    w2vc = W2VClusters(W2V_CLUSTERS_FILEPATH)
    
    # Load the wrapper for the gensim LDA
    print_if_verbose("Loading LDA...")
    lda = LdaWrapper(LDA_FILEPATH, LDA_DICTIONARY_FILEPATH, cache_filepath=LDA_CACHE_FILEPATH)
    
    # Load the wrapper for the stanford POS tagger
    print_if_verbose("Loading POS-Tagger...")
    pos = PosTagger(STANFORD_POS_JAR_FILEPATH, STANFORD_MODEL_FILEPATH, cache_filepath=POS_TAGGER_CACHE_FILEPATH)
    
    # create feature generators
    result = [
        StartsWithUppercaseFeature(),
        TokenLengthFeature(),
        ContainsDigitsFeature(),
        ContainsPunctuationFeature(),
        OnlyDigitsFeature(),
        OnlyPunctuationFeature(),
        W2VClusterFeature(w2vc),
        BrownClusterFeature(bc),
        BrownClusterBitsFeature(bc),
        GazetteerFeature(gaz),
        WordPatternFeature(),
        UnigramRankFeature(ug_all_top),
        PrefixFeature(),
        SuffixFeature(),
        POSTagFeature(pos),
        LDATopicFeature(lda, LDA_WINDOW_LEFT_SIZE, LDA_WINDOW_LEFT_SIZE)
    ]
    
    return result


class StartsWithUppercaseFeature(object):
    """Generates a feature that describes, whether a given token starts with an uppercase letter."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        pass
    
    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["swu=%d" % (int(token.word[:1].istitle()))])
        return result

class TokenLengthFeature(object):
    """Generates a feature that describes the character length of a token."""
    def __init__(self, max_length=30):
        """Instantiates a new object of this feature generator.
        Args:
            max_length: The max length to return in the generated features, e.g. if set to 30 you
                will never get a "l=31" result, only "l=30" for a token with length >= 30.
        """
        self.max_length = max_length

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["l=%d" % (min(len(token.word), self.max_length))])
        return result

class ContainsDigitsFeature(object):    
    """Generates a feature that describes, whether a token contains any digit."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexpContainsDigits = re.compile(r'[0-9]+')

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["cD=%d" % (int(self.regexpContainsDigits.search(token.word) is not None))])
        return result
    
class ContainsPunctuationFeature(object):
    """Generates a feature that describes, whether a token contains any punctuation."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexpContainsPunctuation = re.compile(r'[\.\,\:\;\(\)\[\]\?\!]+')

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["cP=%d" % (int(self.regexpContainsPunctuation.search(token.word) is not None))])
        return result

class OnlyDigitsFeature(object):
    """Generates a feature that describes, whether a token contains only digits."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexpContainsOnlyDigits = re.compile(r'^[0-9]+$')

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["oD=%d" % (int(self.regexpContainsOnlyDigits.search(token.word) is not None))])
        return result

class OnlyPunctuationFeature(object):
    """Generates a feature that describes, whether a token contains only punctuation."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        self.regexpContainsOnlyPunctuation = re.compile(r'^[\.\,\:\;\(\)\[\]\?\!]+$')

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["oP=%d" % (int(self.regexpContainsOnlyPunctuation.search(token.word) is not None))])
        return result

class W2VClusterFeature(object):
    """Generates a feature that describes the word2vec cluster of the token."""
    def __init__(self, w2v_clusters):
        """Instantiates a new object of this feature generator.
        Args:
            w2v_clusters: An instance of W2VClusters as defined in w2v.py that can be queried to
                estimate the cluster of a word.
        """
        self.w2v_clusters = w2v_clusters

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["w2v=%d" % (self.token_to_cluster(token))])
        return result
    
    def token_to_cluster(self, token):
        """Converts a token/word to its cluster index among the word2vec clusters.
        Args:
            token: The token/word to convert.
        Returns:
            cluster index as integer,
            or -1 if it wasn't found among the w2v clusters.
        """
        return self.w2v_clusters.get_cluster_of(token.word, -1)

class BrownClusterFeature(object):
    """Generates a feature that describes the brown cluster of the token."""
    def __init__(self, brown_clusters):
        """Instantiates a new object of this feature generator.
        Args:
            brown_clusters: An instance of BrownClusters as defined in brown.py that can be queried
                to estimate the brown cluster of a word.
        """
        self.brown_clusters = brown_clusters

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["bc=%d" % (self.token_to_cluster(token))])
        return result
    
    def token_to_cluster(self, token):
        """Converts a token/word to its cluster index among the brown clusters.
        Args:
            token: The token/word to convert.
        Returns:
            cluster index as integer,
            or -1 if it wasn't found among the brown clusters.
        """
        return self.brown_clusters.get_cluster_of(token.word, -1)

class BrownClusterBitsFeature(object):
    """Generates a feature that contains the brown cluster bitchain of the token."""
    def __init__(self, brown_clusters):
        """Instantiates a new object of this feature generator.
        Args:
            brown_clusters: An instance of BrownClusters as defined in brown.py that can be queried
                to estimate the brown cluster bitchain of a word.
        """
        self.brown_clusters = brown_clusters

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["bcb=%s" % (self.token_to_bitchain(token)[0:7])])
        return result
    
    def token_to_bitchain(self, token):
        """Converts a token/word to its brown cluster bitchain among the brown clusters.
        Args:
            token: The token/word to convert.
        Returns:
            brown cluster bitchain as string,
            or "" (empty string) if it wasn't found among the brown clusters.
        """
        return self.brown_clusters.get_bitchain_of(token.word, "")

class GazetteerFeature(object):
    """Generates a feature that describes, whether a token is contained in the gazetteer."""
    def __init__(self, gazetteer):
        """Instantiates a new object of this feature generator.
        Args:
            gazetteer: An instance of Gazetteer as defined in gazetteer.py that can be queried
                to estimate whether a word is contained in an Gazetteer.
        """
        self.gazetteer = gazetteer
    
    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["g=%d" % (int(self.is_in_gazetteer(token)))])
        return result
    
    def is_in_gazetteer(self, token):
        """Returns True if the token/word appears in the gazetteer.
        Args:
            token: The token/word.
        Returns:
            True if the word appears in the gazetter, False otherwise.
        """
        return self.gazetteer.contains(token.word)

class WordPatternFeature(object):
    """Generates a feature that describes the word pattern of a feature.
    A word pattern is a rough representation of the word, examples:
        original word | word pattern
        ----------------------------
        John          | Aa+
        Washington    | Aa+
        DARPA         | A+
        2055          | 9+
    """
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        # maximum length of tokens after which to simply cut off
        self.max_length = 15
        # if cut off because of maximum length, use this char at the end of the word to signal
        # the cutoff
        self.max_length_char = "~"
        
        self.normalization = [
            (r"[A-ZÄÖÜ]", "A"),
            (r"[a-zäöüß]", "a"),
            (r"[0-9]", "9"),
            (r"[\.\!\?\,\;]", "."),
            (r"[\(\)\[\]\{\}]", "("),
            (r"[^Aa9\.\(]", "#")
        ]
        
        # note: we do not map numers to 9+, e.g. years will still be 9999
        self.mappings = [
            (r"[A]{2,}", "A+"),
            (r"[a]{2,}", "a+"),
            (r"[\.]{2,}", ".+"),
            (r"[\(]{2,}", "(+"),
            (r"[#]{2,}", "#+")
        ]

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["wp=%s" % (self.token_to_wordpattern(token))])
        return result
    
    def token_to_wordpattern(self, token):
        """Converts a token/word to its word pattern.
        Args:
            token: The token/word to convert.
        Returns:
            The word pattern as string.
        """
        normalized = token.word
        for from_regex, to_str in self.normalization:
            normalized = re.sub(from_regex, to_str, normalized)
        
        wp = normalized
        for from_regex, to_str in self.mappings:
            wp = re.sub(from_regex, to_str, wp)
        
        if len(wp) > self.max_length:
            wp = wp[0:self.max_length] + self.max_length_char
        
        return wp

class UnigramRankFeature(object):
    """Generates a feature that describes the rank of the word among a list of unigrams, ordered
    descending, i.e. the most common word would have the rank 1.
    """
    def __init__(self, unigrams):
        """Instantiates a new object of this feature generator.
        Args:
            unigrams: An instance of Unigrams as defined in unigrams.py that can be queried
                to estimate the rank of a word among all unigrams.
        """
        self.unigrams = unigrams

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            result.append(["ng1=%d" % (self.token_to_rank(token))])
        return result
    
    def token_to_rank(self, token):
        """Converts a token/word to its unigram rank.
        Args:
            token: The token/word to convert.
        Returns:
            Unigram rank as integer,
            or -1 if it wasn't found among the unigrams.
        """
        return self.unigrams.get_rank_of(token.word, -1)

class PrefixFeature(object):
    """Generates a feature that describes the prefix (the first three chars) of the word."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        pass

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            prefix = re.sub(r"[^a-zA-ZäöüÄÖÜß\.\,\!\?]", "#", token.word[0:3])
            result.append(["pf=%s" % (prefix)])
        return result

class SuffixFeature(object):
    """Generates a feature that describes the suffix (the last three chars) of the word."""
    def __init__(self):
        """Instantiates a new object of this feature generator."""
        pass

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for token in window.tokens:
            suffix = re.sub(r"[^a-zA-ZäöüÄÖÜß\.\,\!\?]", "#", token.word[-3:])
            result.append(["sf=%s" % (suffix)])
        return result

class POSTagFeature(object):
    """Generates a feature that describes the Part Of Speech tag of the word."""
    def __init__(self, pos_tagger):
        """Instantiates a new object of this feature generator.
        Args:
            pos_tagger: An instance of PosTagger as defined in pos.py that can be queried
                to estimate the POS-tag of a word.
        """
        self.pos_tagger = pos_tagger

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        pos_tags = self.stanford_pos_tag(window)
        result = []
        for i, token in enumerate(window.tokens):
            word, pos_tag = pos_tags[i][0], pos_tags[i][1]
            result.append(["pos=%s" % (pos_tag)])
        return result
    
    def stanford_pos_tag(self, window):
        """Converts a Window (list of tokens) to their POS tags.
        Args:
            window: Window object containing the token list to POS-tag.
        Returns:
            List of POS tags as strings.
        """
        return self.pos_tagger.tag([token.word for token in window.tokens])

class LDATopicFeature(object):
    """Generates a list of features that contains one or more topics of the window around the
    word."""
    def __init__(self, lda_wrapper, window_left_size, window_right_size, prob_threshold=0.2):
        """Instantiates a new object of this feature generator.
        Args:
            lda_wrapper: An instance of LdaWrapper as defined in models/lda.py that can be queried
                to estimate the LDA topics of a window around a word.
            window_left_size: Size in words/tokens to the left of a word/token to use for the LDA.
            window_right_size: See window_left_size.
            prob_threshold: The probability threshold to use for the topics. If a topic has a
                higher porbability than this threshold, it will be added as a feature,
                e.g. "lda_15=1" if topic 15 has a probability >= 0.2.
        """
        self.lda_wrapper = lda_wrapper
        self.window_left_size = window_left_size
        self.window_right_size = window_right_size
        self.prob_threshold = prob_threshold

    def convert_window(self, window):
        """Converts a Window object into a list of lists of features, where features are strings.
        Args:
            window: The Window object (defined in datasets.py) to use.
        Returns:
            List of lists of features.
            One list of features for each token.
            Each list can contain any number of features (including 0).
            Each feature is a string.
        """
        result = []
        for i, token in enumerate(window.tokens):
            token_features = []
            window_start = max(0, i - self.window_left_size)
            window_end = min(len(window.tokens), i + self.window_right_size + 1)
            window_tokens = window.tokens[window_start:window_end]
            text = " ".join([token.word for token in window_tokens])
            topics = self.get_topics(text)
            for (topic_idx, prob) in topics:
                if prob > self.prob_threshold:
                    token_features.append("lda_%d=%s" % (topic_idx, "1"))
            result.append(token_features)
        return result
    
    def get_topics(self, text):
        """Converts a small text window (string) to its LDA topics.
        Args:
            text: The small text window to convert (as string).
        Returns:
            List of tuples of form (topic index, probability).
        """
        return self.lda_wrapper.get_topics(text)
