# -*- coding: utf-8 -*-
"""
    File to collect all unigrams and all name-unigrams (label PER) from a corpus file.
    The corpus file must have one document/article per line. The words must be labeled in the
    form word/LABEL.
    Example file content:
        Yestarday John/PER Doe/PER said something amazing.
        Washington/LOC D.C./LOC is the capital of the U.S.
        The foobird is a special species of birds. It's commonly found on mars.
        ...

    Execute via:
        python -m preprocessing/collect_unigrams
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from model.unigrams import Unigrams

# All capitalized constants come from this file
import config as cfg

def main():
    """Main function. Gathers all unigrams and name-unigrams, see documantation at the top."""

    # collect all unigrams (all labels, including "O")
    print("Collecting unigrams...")
    ug_all = Unigrams()
    ug_all.fill_from_articles(cfg.ARTICLES_FILEPATH, verbose=True)
    ug_all.write_to_file(cfg.UNIGRAMS_FILEPATH)
    ug_all = None

    # collect only unigrams of label PER
    print("Collecting person names (label=PER)...")
    ug_names = Unigrams()
    ug_names.fill_from_articles_labels(cfg.ARTICLES_FILEPATH, ["PER"], verbose=True)
    ug_names.write_to_file(cfg.UNIGRAMS_PERSON_FILEPATH)

    print("Finished.")

# ---------------

if __name__ == "__main__":
    main()
