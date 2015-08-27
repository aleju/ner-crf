# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

BROWN_CLUSTERS_FILEPATH = ""
UNIGRAMS_NAMES_FILEPATH = ""
UNIGRAMS_FILEPATH = ""
LDA_FILEPATH = ""
LDA_DICTIONARY_FILEPATH = ""
LDA_CACHE_MAX_SIZE = 0
STANFORD_POS_JAR_FILEPATH = ""
STANFORD_MODEL_FILEPATH = ""
POS_TAGGER_CACHE_DIR = None
UNIGRAMS_SKIP_FIRST_N = 0
UNIGRAMS_MAX_COUNT_WORDS = None
W2V_CLUSTERS_FILEPATH = ""
LDA_WINDOW_LEFT_SIZE = 5
LDA_WINDOW_RIGHT_SIZE = 5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("identifier",
                        help="A short name/identifier for your experiment, " \
                             "e.g. 'ex42b_more_dropout'.")
    
    trainer = pycrfsuite.Trainer(verbose=True)
    
    print("Loading examples...")
    features = create_features()
    examples = articles_to_xy(load_articles(ARTICLES_FILEPATH), 50, features, preferLabeledChunks=True)
    
    print("Appending up to %d examples...".format(COUNT_EXAMPLES))
    added = 0
    for (features, labels, tokens) in examples:
        if added > 0 and added % 500 == 0:
            print("Appended %d examples...".format(added))
        trainer.append(features, labels)
        added += 1
        if added == COUNT_EXAMPLES:
            break
    
    print("Training...")
    if MAX_ITERATIONS is not None and MAX_ITERATIONS > 0:
        trainer.set_params({'max_iterations': MAX_ITERATIONS})
    trainer.train(identifier)

def create_features():
    bc = BrownClusters(BROWN_CLUSTERS_FILEPATH)
    gaz = Gazetteer(UNIGRAMS_NAMES_FILEPATH, UNIGRAMS_FILEPATH)
    lda = LdaWrapper(LDA_FILEPATH, LDA_DICTIONARY_FILEPATH, cache_max_size=LDA_CACHE_MAX_SIZE)
    pos = PosTagger(STANFORD_POS_JAR_FILEPATH, STANFORD_MODEL_FILEPATH, cache_dir=POS_TAGGER_CACHE_DIR)
    ug = Unigrams(UNIGRAMS_FILEPATH, skip_first_n=UNIGRAMS_SKIP_FIRST_N, max_count_words=UNIGRAMS_MAX_COUNT_WORDS)
    w2vc = W2VClusters(W2V_CLUSTERS_FILEPATH)
    
    result = [
        StartsWithUppercaseFeature(),
        TokenLengthFeature(),
        ContainsDigitsFeature(),
        ContainsPunctuationFeature(),
        OnlyDigitsFeature(),
        OnlyPunctuationFeature(),
        W2VClusterFeature(w2vc),
        BrownClusterFeature(bc),
        BrownClusterBitsFeature(bc),
        GazetteerFeature(gaz),
        WordPatternFeature(),
        UnigramRankFeature(ug),
        PrefixFeature(),
        SuffixFeature(),
        POSTagFeature(pos),
        LDATopicFeature(lda, LDA_WINDOW_LEFT_SIZE, LDA_WINDOW_LEFT_SIZE)
    ]
    
    return result

if __name__ == "__main__":
    main()
