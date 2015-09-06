# -*- coding: utf-8 -*-
"""
Script to test a trained CRF model.
train.py must be used before this to train the CRF.
This file must be called with the same identifier that was used during training.

Example usage:
    python test.py --identifier="my_experiment" --mycorpus
    python test.py --identifier="my_experiment" --germeval

The first command tests on the corpus set in ARTICLES_FILEPATH.
The second command tests on the germeval corpus, whichs path is defined in GERMEVAL_FILEPATH.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import random
import pycrfsuite
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from model.datasets import load_windows, load_articles, generate_examples, Article
import model.features as features

# All capitalized constants come from this file
import config as cfg

random.seed(42)

def main():
    """Main method to handle command line arguments and then call the testing methods."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", required=True,
                        help="A short name/identifier for your experiment, e.g. 'ex42b'.")
    parser.add_argument("--mycorpus", required=False, action="store_const", const=True,
                        help="Whether to test on your corpus, defined via the constant " \
                             "ARTICLES_FILEPATH.")
    parser.add_argument("--germeval", required=False, action="store_const", const=True,
                        help="Whether to test on the german eval 2014 corpus.")
    args = parser.parse_args()

    # test on corpus set in ARTICLES_FILEPATH
    if args.mycorpus:
        test_on_mycorpus(args)
    # test on germeval corpus
    if args.germeval:
        test_on_germeval(args)
    if not args.mycorpus and not args.germeval:
        print("Expected either --mycorpus or --germeval flag")

def test_on_mycorpus(args):
    """Tests on the corpus set in ARTICLES_FILEPATH.
    Prints a full report, including precision, recall and F1 score per label.

    Args:
        args: Command line arguments as parsed by argparse.ArgumentParser.
    """
    print("Testing on mycorpus (%s)..." % (cfg.ARTICLES_FILEPATH))
    test_on_articles(args.identifier, load_articles(cfg.ARTICLES_FILEPATH),
                     nb_append=cfg.COUNT_WINDOWS_TEST)

def test_on_germeval(args):
    """Tests on the germeval corpus.
    The germeval filepath is defined in GERMEVAL_FILEPATH.
    See https://sites.google.com/site/germeval2014ner/data .

    Args:
        args: Command line arguments as parsed by argparse.ArgumentParser.
    """
    print("Testing on germeval (%s)..." % (cfg.GERMEVAL_FILEPATH))
    test_on_articles(args.identifier, load_germeval(cfg.GERMEVAL_FILEPATH))

def test_on_articles(identifier, articles, nb_append=None):
    """Test a trained CRF model on a list of Article objects (annotated text).

    Will print a full classification report by label (f1, precision, recall).

    Args:
        identifier: Identifier of the trained model to be used.
        articles: A list of Article objects or a generator for such a list. May only contain
            one single Article object.
    """
    print("Loading tagger...")
    tagger = pycrfsuite.Tagger()
    tagger.open(identifier)

    # create feature generators
    # this may take a while
    print("Creating features...")
    feature_generators = features.create_features()

    # create window generator
    print("Loading windows...")
    windows = load_windows(articles, cfg.WINDOW_SIZE, feature_generators, only_labeled_windows=True)

    # load feature lists and label lists (X, Y)
    # this may take a while
    all_feature_values_lists = []
    correct_label_chains = []
    for fvlist, labels in generate_examples(windows, nb_append=nb_append):
        all_feature_values_lists.append(fvlist)
        correct_label_chains.append(labels)

    # generate predicted chains of labels
    print("Testing on %d windows..." % (len(all_feature_values_lists)))
    predicted_label_chains = [tagger.tag(fvlists) for fvlists in all_feature_values_lists]

    # print classification report (precision, recall, f1)
    print(bio_classification_report(correct_label_chains, predicted_label_chains))

def load_germeval(filepath):
    """Loads the source of the gereval 2014 corpus and converts it to a list of Article objects.

    Args:
        filepath: Filepath to the source file, e.g. "/var/foo/NER-de-test.tsv".
    Returns:
        List of Article
        (will contain only one single Article object).
    """
    lines = open(filepath, "r").readlines()
    lines = [line.decode("utf-8").strip() for line in lines]
    # remove lines that are comments
    lines = [line for line in lines if line[0:1] != "#"]
    # remove all empty lines
    lines = [line for line in lines if len(line) > 0]

    sentence = []
    sentences = []

    for line_idx, line in enumerate(lines):
        blocks = line.split("\t")
        (number, word, tag1, _) = blocks # 4th block would be tag2
        number = int(number)

        # if we reach the next sentence, add the previous sentence to the 'sentences' container
        if (number == 1 and len(sentence) > 0) or line_idx == len(lines) - 1:
            sentences.append(sentence)
            sentence = []

        # convert all labels containing OTH (OTHER) so MISC
        if "OTH" in tag1:
            tag1 = "MISC"

        # Add the word in an annotated way if the tag1 looks like one of the labels in the
        # allowed labels (config setting LABELS). We don't check for full equality here, because
        # that allows BIO tags (e.g. B-PER) to also be accepted. They will automatically be
        # normalized by the Token objects (which will also throw away unnormalizable annotations).
        # Notice that we ignore tag2 as tag1 is usually the more important one.
        contains_label = any([(label in tag1) for label in cfg.LABELS])
        is_blacklisted = any([(bl_label in tag1) for bl_label in ["part", "deriv"]])
        if contains_label and not is_blacklisted:
            sentence.append(word + "/" + tag1)
        else:
            sentence.append(word)

    return [Article(" ".join(sentence)) for sentence in sentences]

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!

    Note: This function was copied from
    http://nbviewer.ipython.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb

    Args:
        y_true: True labels, list of strings
        y_pred: Predicted labels, list of strings
    Returns:
        classification report as string
    """
    lbin = LabelBinarizer()
    y_true_combined = lbin.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lbin.transform(list(chain.from_iterable(y_pred)))

    #tagset = set(lbin.classes_) - {NO_NE_LABEL}
    tagset = set(lbin.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lbin.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )

# ----------------------

if __name__ == "__main__":
    main()
