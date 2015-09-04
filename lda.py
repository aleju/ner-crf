# -*- coding: utf-8 -*
from __future__ import absolute_import, division, print_function, unicode_literals
import gensim
#from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from model.datasets import load_articles, load_windows
import sys
import argparse

#ARTICLES_FILEPATH = "/media/ssd2/nlp/corpus/processed/wikipedia-de/all.txt"
ARTICLES_FILEPATH = "/media/aj/grab/nlp/corpus/processed/wikipedia-ner/annotated-fulltext.txt"
PER_EXAMPLE_WINDOW_SIZE = 11
LDA_CHUNK_SIZE = 10000 #2000 * 100  # docs pro batch in LDA, default ist 2000
COUNT_EXAMPLES_FOR_DICTIONARY = 100000
#COUNT_EXAMPLES_FOR_LDA = 1000 * 1000 * 10 # in windows
COUNT_EXAMPLES_FOR_LDA = 1000 * 1000 # in windows
LDA_COUNT_TOPICS = 100
LDA_COUNT_WORKERS = 3
LDA_MODEL_FILENAME = "lda_model"
LDA_DICTIONARY_FILENAME = "lda_dictionary"
IGNORE_WORDS_BELOW_COUNT = 4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict", required=False, action="store_const", const=True,
                        help="Create the LDA's dictionary (must happen before training).")
    parser.add_argument("--train", required=False, action="store_const", const=True,
                        help="Train the LDA model.")
    args = parser.parse_args()

    if args.dict:
        generate_dictionary()
    if args.train:
        train_lda()
    if not args.dict and not args.train:
        print("No option chosen.")
        print("Add --dict to generate the dictionary or --train to train the LDA model.")

def generate_dictionary():
    print("------------------")
    print("Generating LDA Dictionary")
    print("------------------")
    
    articles = load_articles(ARTICLES_FILEPATH)
    articles_str = []
    dictionary = gensim.corpora.Dictionary()
    update_every_n_articles = 1000

    for i, article in enumerate(articles):
        articles_str.append(article.get_content_as_string().lower().split(" "))
        if len(articles_str) >= update_every_n_articles:
            print("Updating (at article %d of max %d)..." % (i, COUNT_EXAMPLES_FOR_DICTIONARY))
            dictionary.add_documents(articles_str)
            articles_str = []
        
        if i > COUNT_EXAMPLES_FOR_DICTIONARY:
            print("Reached max of %d articles." % (COUNT_EXAMPLES_FOR_DICTIONARY,))
            break

    if len(articles_str) > 0:
        print("Updating with remaining articles...")
        dictionary.add_documents(articles_str)

    print("Loaded %d unique words." % (len(dictionary.keys()),))

    print("Filtering rare words...")
    rare_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq < IGNORE_WORDS_BELOW_COUNT]
    dictionary.filter_tokens(rare_ids)
    dictionary.compactify()

    print("Filtered to %d unique words." % (len(dictionary.keys()),))
    print("Saving dictionary...")
    dictionary.save(LDA_DICTIONARY_FILENAME)

def train_lda():
    print("------------------")
    print("Training LDA model")
    print("------------------")
    
    print("Loading dictionary...")
    dictionary = gensim.corpora.dictionary.Dictionary.load(LDA_DICTIONARY_FILENAME)

    print("Generating id2word...")
    id2word = {}
    for word in dictionary.token2id:    
        id2word[dictionary.token2id[word]] = word

    #corpus = [dictionary.doc2bow(text) for text in texts]

    #gensim.corpora.MmCorpus.serialize('wikipedia_lda_corpus.mm', corpus)
    #mm = gensim.corpora.MmCorpus('wikipedia_lda_corpus.mm')

    print("Training...")
    lda_model = LdaMulticore(corpus=None, num_topics=LDA_COUNT_TOPICS, id2word=id2word, workers=LDA_COUNT_WORKERS, chunksize=LDA_CHUNK_SIZE)

    examples = []
    update_every_n_windows = 100000
    windows = load_windows(load_articles(ARTICLES_FILEPATH), PER_EXAMPLE_WINDOW_SIZE, only_labeled_windows=True)
    for i, window in enumerate(windows):
        tokens_str = [token.word.lower() for token in window.tokens]
        bow = dictionary.doc2bow(tokens_str)
        examples.append(bow)
        if len(examples) >= update_every_n_windows:
            print("Updating (at window %d of max %d)..." % (i, COUNT_EXAMPLES_FOR_LDA))
            lda_model.update(examples)
            examples = []
        if i >= COUNT_EXAMPLES_FOR_LDA:
            print("Reached max of %d windows." % (COUNT_EXAMPLES_FOR_LDA,))
            break

    if len(examples) > 0:
        print("Updating with remaining windows...")
        lda_model.update(examples)

    print("Saving...")
    lda_model.save(LDA_MODEL_FILENAME)

    """
    print "Showing Topics..."
    topics = wpLda.show_topics(num_topics=LDA_COUNT_TOPICS, num_words=10, log=False, formatted=True)

    print "Showing Examples..."
    for (i, topic) in enumerate(topics):
        print str(i) + ": " + topic
    print wpLda[dictionary.doc2bow(u'der Schriftsteller'.lower().split(" "))]
    print wpLda[dictionary.doc2bow(u'der portugiesische Staats­präsident Jorge Sampaio wie am'.lower().split(" "))]
    print wpLda[dictionary.doc2bow(u'Harold Johnson ( 86 ), US-amerikanischer Boxer ( † 19 . Februar )'.lower().split(" "))]
    print wpLda[dictionary.doc2bow(u'Der kirgisische Fünftausender Pik Alexander von Humboldt wurde 2003'.lower().split(" "))]
    print wpLda[dictionary.doc2bow(u'Textil­unternehmer und Sozial­reformer Bernhard Greuter kommt zur Welt'.lower().split(" "))]
    print wpLda[dictionary.doc2bow(u'In Wien stirbt Joseph II . , seit 1765 Kaiser'.lower().split(" "))]
    print "---"
    print wpLda[dictionary.doc2bow(u'Die NASA-Mond­sonde Ranger 8 funkt , wie'.lower().split(" "))]
    print wpLda[dictionary.doc2bow(u'vermutlich terroristisch motivierten Anschlägen in Kopenhagen wurden'.lower().split(" "))]
    print wpLda[dictionary.doc2bow(u'Gemäß dem 2. Minsker Abkommen ist eine'.lower().split(" "))]
    """

"""
def examplesGen(startAtArticle=0, examplesFilePath=EXAMPLES_FILE_PATH):
	skippedArticlesSoFar = 0
	processedArticlesSoFar = 0
	
	with open(examplesFilePath, 'rU') as f:
		for article in f:
			article = article.decode("utf-8").strip()
			
			if len(article) > 0:
				if skippedArticlesSoFar < startAtArticle:
					skippedArticlesSoFar += 1
				else:
					tokens = article.lower().strip().split(" ")
					yield tokens
					#tokenChunks = chunks(tokens, PER_EXAMPLE_CHUNK_SIZE)
					#for tokenChunk in tokenChunks:
					#	yield tokenChunk
					
def chunks(of, chunkSize):
    for i in xrange(0, len(of), chunkSize):
        yield of[i:i + chunkSize]


class MyCorpus(object):
	dictionary = None
	examplesFilePath = None
	loaded = 0
	
	def __init__(self, dictionary, examplesFilePath=EXAMPLES_FILE_PATH):
		self.dictionary = dictionary
		self.examplesFilePath = examplesFilePath
		
	def __iter__(self):
		with open(self.examplesFilePath, 'rU') as f:
			for article in f:
				article = article.decode("utf-8").strip()
				if len(article) > 0:
					if self.loaded < 50000:
						if self.loaded % 500 == 0:
							print "Loaded " + str(self.loaded) + " articles..."
						self.loaded += 1
						
						tokens = article.lower().strip().split(" ")
						tokenChunks = chunks(tokens, PER_EXAMPLE_CHUNK_SIZE)
						for tokenChunk in tokenChunks:
							yield self.dictionary.doc2bow(tokenChunk)
						# assume there's one document per line, tokens separated by whitespace
						#yield self.dictionary.doc2bow(article.lower().split())
"""

# --------------------

if __name__ == "__main__":
    main()
