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

from model.datasets import load_windows, load_articles, generate_examples
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
    feature_generators = features.create_features()
    
    # Initialize the window generator
    # each window has a fixed maximum size of tokens
    print("Loading windows...")
    windows = load_windows(load_articles(ARTICLES_FILEPATH), WINDOW_SIZE, feature_generators, only_labeled_windows=True)
    
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

# ----------------

if __name__ == "__main__":
    main()
