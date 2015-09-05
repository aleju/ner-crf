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

random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ARTICLES_FILEPATH = "/media/aj/grab/nlp/corpus/processed/wikipedia-ner/annotated-fulltext.txt"
BROWN_CLUSTERS_FILEPATH = "/media/aj/ssd2a/nlp/corpus/brown/wikipedia-de/brown_c1000_min12/paths"
#UNIGRAMS_FILEPATH = "/media/aj/ssd2a/nlp/corpus/processed/wikipedia-de/ngrams-1.txt"
UNIGRAMS_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/unigrams.txt")
UNIGRAMS_PERSON_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/unigrams_per.txt")
LDA_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/lda_model")
LDA_DICTIONARY_FILEPATH = os.path.join(CURRENT_DIR, "preprocessing/lda_dictionary")
LDA_CACHE_MAX_SIZE = 100000
STANFORD_DIR = "/media/aj/ssd2a/nlp/nlpjava/stanford-postagger-full-2013-06-20/stanford-postagger-full-2013-06-20/"
STANFORD_POS_JAR_FILEPATH = os.path.join(STANFORD_DIR, "stanford-postagger-3.2.0.jar")
STANFORD_MODEL_FILEPATH = os.path.join(STANFORD_DIR, "models/german-fast.tagger")
POS_TAGGER_CACHE_FILEPATH = os.path.join(CURRENT_DIR, "pos.cache")
UNIGRAMS_SKIP_FIRST_N = 0
UNIGRAMS_MAX_COUNT_WORDS = 1000
W2V_CLUSTERS_FILEPATH = "/media/aj/ssd2a/nlp/corpus/word2vec/wikipedia-de/classes1000_cbow0_size300_neg0_win10_sample1em3_min50.txt"
WINDOW_SIZE = 50
SKIPCHAIN_LEFT = 5
SKIPCHAIN_RIGHT = 5
LDA_WINDOW_LEFT_SIZE = 5
LDA_WINDOW_RIGHT_SIZE = 5
MAX_ITERATIONS = None
COUNT_EXAMPLES = 100000

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
    
    print("Adding example windows (up to max %d)..." % (COUNT_EXAMPLES))
    added = 0
    for window in windows:
        labels = window.get_labels()
        feature_values_lists = []
        for word_idx in range(len(window.tokens)):
            feature_values_lists.append(window.get_feature_values_list(word_idx, SKIPCHAIN_LEFT, SKIPCHAIN_RIGHT))
        trainer.append(feature_values_lists, labels)
        added += 1
        if added % 500 == 0:
            print("Added %d of max %d windows" % (added, COUNT_EXAMPLES))
        if added == COUNT_EXAMPLES:
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
    lda = LdaWrapper(LDA_FILEPATH, LDA_DICTIONARY_FILEPATH, cache_max_size=LDA_CACHE_MAX_SIZE)
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
