

def main(modelName="ner-crf"):
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
    trainer.train(modelName)



if __name__ == "__main__":
    main()
