# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import argparse
import random
import pycrfsuite
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from model.brown import BrownClusters
from model.gazetteer import Gazetteer
from model.lda import LdaWrapper
from model.pos import PosTagger
from model.unigrams import Unigrams
from model.w2v import W2VClusters
from model.datasets import load_windows, load_articles
import model.features as features

from train import create_features

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
COUNT_EXAMPLES = 1000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", required=True,
                        help="A short name/identifier for your experiment, e.g. 'ex42b'.")
    args = parser.parse_args()
    
    print("Loading tagger...")
    tagger = pycrfsuite.Tagger()
    tagger.open(args.identifier)

    print("Creating features...")
    features = create_features()
    
    print("Loading windows...")
    windows = load_windows(load_articles(ARTICLES_FILEPATH), WINDOW_SIZE, features, only_labeled_windows=True)
    
    correct_label_chains = []
    all_feature_values_lists = []
    added = 0
    for window in windows:
        labels = window.get_labels()
        feature_values_lists = []
        for word_idx in range(len(window.tokens)):
            feature_values_lists.append(window.get_feature_values_list(word_idx, SKIPCHAIN_LEFT, SKIPCHAIN_RIGHT))
        all_feature_values_lists.append(feature_values_lists)
        correct_label_chains.append(labels)
        
        added += 1
        if added % 500 == 0:
            print("Added %d of max %d windows" % (added, COUNT_EXAMPLES))
        if added == COUNT_EXAMPLES:
            break
    
    print("Testing on %d windows..." % (len(all_feature_values_lists)))
    predicted_label_chains = [tagger.tag(fvlists) for fvlists in all_feature_values_lists]
    print(bio_classification_report(correct_label_chains, predicted_label_chains))

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    #tagset = set(lb.classes_) - {NO_NE_LABEL}
    tagset = set(lb.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

if __name__ == "__main__":
    main()
