# -*- coding: utf-8 -*-
"""
Main training file for the CRF.
This file trains a CRF model and saves it under the filename provided via an 'identifier' command
line argument.

Usage example:
    python train.py --identifier="my_experiment"
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import argparse
import random
import pycrfsuite

from model.brown import BrownClusters
from model.gazetteer import Gazetteer
from model.lda import LdaWrapper
from model.pos import PosTagger
from model.unigrams import Unigrams
from model.w2v import W2VClusters
from model.datasets import load_windows, load_articles
import model.features as features

# All capitalized constants come from this file
from config import *

random.seed(42)

def main():
    """This function handles the command line arguments and then calls the train() method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", required=True,
                        help="A short name/identifier for your experiment, e.g. 'ex42b'.")
    args = parser.parse_args()
    
    train(args)

def train(args):
    """Main training method.
    
    Does the following:
        1. Create a new pycrfsuite trainer object. We will have to add feature chains and label
           chains to that object and then train on them.
        2. Creates the feature (generators). A feature generator might e.g. take in a window
           of N tokens and then return ["upper=1"] for each token that starts with an uppercase
           letter and ["upper=0"] for each token that starts with a lowercase letter. (Lists,
           because a token can be converted into multiple features by a single feature generator,
           e.g. the case for LDA as a token may be part of multiple topics.)
        3. Loads windows from the corpus. Each window has a fixed (maximum) size in tokens.
           We only load windows that contain at least one label (named entity), so that we don't
           waste too much time on windows without any label.
        4. Generate features for each chain of tokens (window). That's basically described in (2.).
           Each chain of tokens from a window will be converted to a list of lists.
           One list at the top level representing each token, then another list for the feature
           values. E.g.
             [["w2v=123", "bc=742", "upper=0"], ["w2v=4", "bc=12", "upper=1", "lda4=1"]]
           for two tokens.
        5. Add feature chains and label chains to the trainer.
        6. Train. This may take several hours for 20k windows.
    
    Args:
        args: Command line arguments as parsed by argparse.ArgumentParser.
    """
    trainer = pycrfsuite.Trainer(verbose=True)
    
    # Create/Initialize the feature generators
    # this may take a few minutes
    print("Creating features...")
    features = create_features()
    
    # Initialize the window generator
    # each window has a fixed maximum size of tokens
    print("Loading windows...")
    windows = load_windows(load_articles(ARTICLES_FILEPATH), WINDOW_SIZE, features, only_labeled_windows=True)
    
    # Add chains of features (each list of lists of strings)
    # and chains of labels (each list of strings)
    # to the trainer.
    # This may take a long while, especially because of the lengthy POS tagging.
    # POS tags and LDA results are cached, so the second run through this part will be significantly
    # faster.
    print("Adding example windows (up to max %d)..." % (COUNT_WINDOWS_TRAIN))
    examples = generate_examples(windows, nb_append=COUNT_WINDOWS_TRAIN, nb_skip=COUNT_WINDOWS_TEST,
                                 verbose=True)
    for feature_values_lists, labels in examples:
        trainer.append(feature_values_lists, labels)
    
    # Train the model
    # this may take several hours
    print("Training...")
    if MAX_ITERATIONS is not None and MAX_ITERATIONS > 0:
        # set the maximum number of iterations of defined in the config file
        # the optimizer stops automatically after some iterations if this is not set
        trainer.set_params({'max_iterations': MAX_ITERATIONS})
    trainer.train(args.identifier)

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
                    print("Added %d windows" % (added, nb_append))
                else:
                    print("Added %d of max %d windows" % (added, nb_append))
            if nb_append is not None and added == nb_append:
                break

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
        features.StartsWithUppercaseFeature(),
        features.TokenLengthFeature(),
        features.ContainsDigitsFeature(),
        features.ContainsPunctuationFeature(),
        features.OnlyDigitsFeature(),
        features.OnlyPunctuationFeature(),
        features.W2VClusterFeature(w2vc),
        features.BrownClusterFeature(bc),
        features.BrownClusterBitsFeature(bc),
        features.GazetteerFeature(gaz),
        features.WordPatternFeature(),
        features.UnigramRankFeature(ug_all_top),
        features.PrefixFeature(),
        features.SuffixFeature(),
        features.POSTagFeature(pos),
        features.LDATopicFeature(lda, LDA_WINDOW_LEFT_SIZE, LDA_WINDOW_LEFT_SIZE)
    ]
    
    return result

# ----------------

if __name__ == "__main__":
    main()
