# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import argparse
import random
import pycrfsuite
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

"""
from model.brown import BrownClusters
from model.gazetteer import Gazetteer
from model.lda import LdaWrapper
from model.pos import PosTagger
from model.unigrams import Unigrams
from model.w2v import W2VClusters
"""
from model.datasets import load_windows, load_articles, Article
#import model.features as features

from train import create_features
from config import *

random.seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", required=True,
                        help="A short name/identifier for your experiment, e.g. 'ex42b'.")
    parser.add_argument("--mycorpus", required=False, action="store_const", const=True,
                        help="Whether to test on your corpus, defined via the constant ARTICLES_FILEPATH.")
    parser.add_argument("--gereval", required=False, action="store_const", const=True,
                        help="Whether to test on the german eval corpus.")
    args = parser.parse_args()
    
    if args.mycorpus:
        test_on_mycorpus(args)
    if args.gereval:
        test_on_gereval(args)
    if not args.mycorpus and not args.gereval:
        print("Expected either --mycorpus or --gereval flag")

def test_on_mycorpus(args):
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
        # in train.py we skipped the first COUNT_WINDOWS_TEST windows (i.e. excluded them from
        # training), so here we can just use them
        labels = window.get_labels()
        feature_values_lists = []
        for word_idx in range(len(window.tokens)):
            feature_values_lists.append(window.get_feature_values_list(word_idx, SKIPCHAIN_LEFT, SKIPCHAIN_RIGHT))
        all_feature_values_lists.append(feature_values_lists)
        correct_label_chains.append(labels)
        
        added += 1
        if added % 500 == 0:
            print("Added %d of max %d windows" % (added, COUNT_WINDOWS_TEST))
        if added == COUNT_WINDOWS_TEST:
            break
    
    print("Testing on %d windows..." % (len(all_feature_values_lists)))
    predicted_label_chains = [tagger.tag(fvlists) for fvlists in all_feature_values_lists]
    print(bio_classification_report(correct_label_chains, predicted_label_chains))

def test_on_gereval(args):
    print("Loading tagger...")
    tagger = pycrfsuite.Tagger()
    tagger.open(args.identifier)

    print("Creating features...")
    features = create_features()
    
    print("Loading windows...")
    windows = load_windows(load_gereval(GEREVAL_FILEPATH), WINDOW_SIZE, features, only_labeled_windows=True)

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
            print("Added %d windows" % (added))
    
    print("Testing on %d windows..." % (len(all_feature_values_lists)))
    predicted_label_chains = [tagger.tag(fvlists) for fvlists in all_feature_values_lists]
    print(bio_classification_report(correct_label_chains, predicted_label_chains))

def load_gereval(filepath):
    lines = open(filepath, "r").readlines()
    lines = [line.decode("utf-8").strip() for line in lines]
    # remove lines that are comments
    lines = filter(lambda line: line[0:1] != "#", lines)
    # remove all empty lines
    lines = filter(lambda line: len(line) > 0, lines)

    sentence  = []
    sentences = []

    for line in lines:
        blocks = line.split("\t")
        (number, word, tag1, tag2) = blocks
        number = int(number)
        
        if number == 1 and len(sentence) > 0:
            #sentences.append(sentence)
            sentences.extend(sentence)
            sentence = []
        
        if "OTH" in tag1:
            tag1 = "MISC"
        
        if any([label in tag1 for label in LABELS]):
            sentence.append(word + "/" + tag1)
        else:
            sentence.append(word)

    return [Article(" ".join(sentences))]

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
