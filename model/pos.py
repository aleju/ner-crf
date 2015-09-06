# -*- coding: utf-8 -*-
"""Class that wraps the Stanford POS tagger."""
from __future__ import absolute_import, division, print_function, unicode_literals
import nltk
import shelve
import random

class PosTagger(object):
    """Class that wraps the Stanford POS tagger.

    This class uses a shelve cache to store generated results. This speeds up the generation
    of training examples, if the identical corpus, window sizes etc. are used.
    """
    def __init__(self, stanford_postagger_jar_filepath, stanford_model_filepath,
                 cache_filepath=None):
        """Initialize the Stanford POS tag wrapper.
        Args:
            stanford_postagger_jar_filepath: Filepath to the jar of the stanford tagger,
                e.g. "/var/foo/bar/stanford-pos-tagger/stanford-postagger-3.2.0.jar".
            stanford_model_filepath: Filepath to the used model for the pos tagger,
                e.g. "/var/foo/bar/stanford-pos-tagger/models/german-fast.tagger".
            cache_filepath: Optional filepath to a shelve cache for the LDA results.
        """
        self.max_string_length = 2000
        self.min_string_length = 1

        self.tagger = nltk.tag.stanford.StanfordPOSTagger(stanford_model_filepath,
                                                          stanford_postagger_jar_filepath,
                                                          encoding="utf-8")

        self.cache_synch_prob = 2 # in percent, 1 to 100
        self.cache_filepath = cache_filepath
        self.cache = shelve.open(cache_filepath) if cache_filepath is not None else None

    def tag(self, tokens):
        """Annotate a list of strings with their POS tags.
        Args:
            tokens: List of strings.
        Returns:
            List of strings (POS tags)
        """
        if self.cache is None:
            return self.tag_uncached(tokens)
        else:
            text = " ".join(tokens)
            _hash = str(hash(text))
            if self.cache.has_key(_hash):
                return self.cache[_hash]
            else:
                tagged = self.tag_uncached(tokens)

                self.cache[_hash] = tagged
                if random.randint(1, 100) <= self.cache_synch_prob:
                    self.synchronize_cache()

                return tagged

    def tag_uncached(self, tokens):
        """Annotate a list of strings with their POS tags without querying the cache.
        Args:
            tokens: List of strings.
        Returns:
            List of strings (POS tags)
        """
        # length of each word + count of required whitespaces between each word
        # max() to avoid -1 if the list of tokens in empty
        total_length = sum([len(token) for token in tokens]) + (max(len(tokens) - 1, 0))
        if total_length >= self.max_string_length:
            raise Exception("String to POS-tag is too long (%d vs max " \
                            "%d)." % (total_length, self.max_string_length))
        elif total_length < self.min_string_length:
            raise Exception("String to POS-tag is too short (%d vs min "\
                            "%d)." % (total_length, self.min_string_length))

        return self.tagger.tag(tokens)

    def synchronize_cache(self):
        """Synchronizes the shelve cache on the HDD with the version in the RAM."""
        self.cache.sync()
