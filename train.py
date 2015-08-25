# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("identifier",
                        help="A short name/identifier for your experiment, " \
                             "e.g. 'ex42b_more_dropout'.")
    
    trainer = pycrfsuite.Trainer(verbose=True)
    
    print("Loading examples...")
    examples = load_xy(preferLabeledChunks=True)
    
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
    bc = BrownClusters(filepath)
    gaz = Gazetteer(unigrams_names_filepath, unigrams_filepath)
    lda = LdaWrapper(lda_filepath, dictionary_filepath, cache_max_size=0)
    pos = PosTagger(stanford_postagger_jar_filepath, stanford_model_filepath, cache_dir=None)
    ug = Unigrams(filepath, skip_first_n=0, max_count_words=None)
    w2vc = W2VClusters(filepath)
    
    result = [
        StartsWithUppercase(),
        TokenLength(),
        ContainsDigits(),
        ContainsPunctuation(),
        OnlyDigits(),
        OnlyPunctuation(),
        W2VCluster(),
        BrownCluster(),
        BrownClusterBits(),
        Gazzetteer(),
        WordPattern(),
        UnigramRank(),
        Prefix(),
        Suffix(),
        POSTag(),
        LDATopic()
    ]
    return result

if __name__ == "__main__":
    main()
