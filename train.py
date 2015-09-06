# -*- coding: utf-8 -*-
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

from config import *

random.seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", required=True,
                        help="A short name/identifier for your experiment, e.g. 'ex42b'.")
    args = parser.parse_args()
    
    trainer = pycrfsuite.Trainer(verbose=True)
    
    print("Creating features...")
    features = create_features()
    
    print("Loading windows...")
    windows = load_windows(load_articles(ARTICLES_FILEPATH), WINDOW_SIZE, features, only_labeled_windows=True)
    
    print("Adding example windows (up to max %d)..." % (COUNT_WINDOWS_TRAIN))
    skipped = 0
    added = 0
    for window in windows:
        # skip the first COUNT_WINDOWS_TEST windows, because we are going to use them
        # to test our trained model
        if skipped < COUNT_WINDOWS_TEST:
            skipped += 1
        else:
            labels = window.get_labels()
            feature_values_lists = []
            for word_idx in range(len(window.tokens)):
                fvl = window.get_feature_values_list(word_idx, SKIPCHAIN_LEFT, SKIPCHAIN_RIGHT)
                #print([token.word for token in window.tokens])
                #print(word_idx)
                #print(labels)
                #print(fvl)
                feature_values_lists.append(fvl)
            trainer.append(feature_values_lists, labels)
            added += 1
            if added % 500 == 0:
                print("Added %d of max %d windows" % (added, COUNT_WINDOWS_TRAIN))
            if added == COUNT_WINDOWS_TRAIN:
                break
    
    print("Training...")
    if MAX_ITERATIONS is not None and MAX_ITERATIONS > 0:
        trainer.set_params({'max_iterations': MAX_ITERATIONS})
    trainer.train(args.identifier)

def create_features(verbose=True):
    print("Loading unigrams...")
    ug_all_top = Unigrams(UNIGRAMS_FILEPATH, skip_first_n=UNIGRAMS_SKIP_FIRST_N, max_count_words=UNIGRAMS_MAX_COUNT_WORDS)
    ug_all = Unigrams(UNIGRAMS_FILEPATH)
    print("Loading person name unigrams...")
    ug_names = Unigrams(UNIGRAMS_PERSON_FILEPATH)
    #ug_names.fill_names_from_file(ARTICLES_FILEPATH, ["PER"])
    print("Creating gazetteer...")
    gaz = Gazetteer(ug_names, ug_all)
    ug_all = None
    ug_names = None
    
    print("Loading brown clusters...")
    bc = BrownClusters(BROWN_CLUSTERS_FILEPATH)
    print("Loading W2V clusters...")
    w2vc = W2VClusters(W2V_CLUSTERS_FILEPATH)
    print("Loading LDA...")
    lda = LdaWrapper(LDA_FILEPATH, LDA_DICTIONARY_FILEPATH, cache_filepath=LDA_CACHE_FILEPATH)
    print("Loading POS-Tagger...")
    pos = PosTagger(STANFORD_POS_JAR_FILEPATH, STANFORD_MODEL_FILEPATH, cache_filepath=POS_TAGGER_CACHE_FILEPATH)
    
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

if __name__ == "__main__":
    main()
