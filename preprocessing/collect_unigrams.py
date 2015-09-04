# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from model.unigrams import Unigrams

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ARTICLES_FILEPATH = "/media/aj/grab/nlp/corpus/processed/wikipedia-ner/annotated-fulltext.txt"
WRITE_UNIGRAMS_FILEPATH = os.path.join(CURRENT_DIR, "unigrams.txt")
WRITE_UNIGRAMS_PERSON_FILEPATH = os.path.join(CURRENT_DIR, "unigrams_per.txt")

def main():
    print("Collecting unigrams...")
    ug_all = Unigrams()
    ug_all.fill_from_articles(ARTICLES_FILEPATH, verbose=True)
    ug_all.write_to_file(WRITE_UNIGRAMS_FILEPATH)
    ug_all = None
    
    print("Collecting person names (label=PER)...")
    ug_names = Unigrams()
    ug_names.fill_from_articles_labels(ARTICLES_FILEPATH, ["PER"], verbose=True)
    ug_names.write_to_file(WRITE_UNIGRAMS_PERSON_FILEPATH)
    
    print("Finished.")

if __name__ == "__main__":
    main()
