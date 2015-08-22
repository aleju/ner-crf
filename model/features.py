class StartsWithUppercase(object):
    def __init__(self):
        pass

class TokenLength(object):
    def __init__(self):
        pass

    if "l" in activeFeatures:
        features.append('l=' + str(min(len(word), 30)))

class ContainsDigits(object):    
    def __init__(self):
        pass

    # contains digits
    if "cD" in activeFeatures:
        features.append('cD=' + str(int(regexpContainsDigits.search(word) is not None)))
    
class ContainsPunctuation(object):
    def __init__(self):
        pass

    # contains punctuation
    if "cP" in activeFeatures:
        features.append('cP=' + str(int(regexpContainsPunctuation.search(word) is not None)))

class OnlyDigits(object):
    def __init__(self):
        pass

    # contains only digits
    if "oD" in activeFeatures:
        features.append('oD=' + str(int(regexpContainsOnlyDigits.search(word) is not None)))

class OnlyPunctuation(object):
    def __init__(self):
        pass

    # contains only punctuation
    if "oP" in activeFeatures:
        features.append('oP=' + str(int(regexpContainsOnlyPunctuation.search(word) is not None)))

class W2VCluster(object):
    def __init__(self):
        pass

    # w2v cluster
    if "w2v" in activeFeatures:
        features.append('w2v=' + str(w2vCluster))

class BrownCluster(object):
    def __init__(self):
        pass

    # brown cluster
    if "bc" in activeFeatures:
        features.append('bc=' + str(brownCluster))

class BrownClusterBits(object):
    def __init__(self):
        pass

    # brown cluster bits
    if "bcb" in activeFeatures:
        features.append('bcb=' + str(brownClusterBits[0:7]))

class Gazzetteer(object):
    def __init__(self):
        pass

    # is in gazetta
    if "g" in activeFeatures:
        features.append('g=' + str(isInGazetta))

class WordPattern(object):
    def __init__(self):
        pass

    # word pattern
    if "wp" in activeFeatures:
        features.append('wp=' + str(wordPattern))

class UnigramRank(object):
    def __init__(self):
        pass

    # unigram pos
    if "ng1" in activeFeatures:
        features.append('ng1=' + str(unigramPos))

class Prefix(object):
    def __init__(self):
        pass

    # prefix
    if "pf" in activeFeatures:
        features.append('pf=' + str(re.sub(r'[^a-zA-ZäöüÄÖÜß\.\,\!\?]', '#', word[0:3])))

class Suffix(object):
    def __init__(self):
        pass

    # suffix
    if "sf" in activeFeatures:
        features.append('sf=' + str(re.sub(r'[^a-zA-ZäöüÄÖÜß\.\,\!\?]', '#', word[-3:])))

class POSTag(object):
    def __init__(self):
        pass

    # pos tag
    if "pos" in activeFeatures:
        features.append('pos=' + str(posTags[i]))

class LDATopic(object):
    def __init__(self):
        pass

    # LDA
    if "lda" in activeFeatures:
        startLda = max(0,         i - LDA_WINDOW_LEFT_WORDS)
        endLda   = min(len(sent), i + LDA_WINDOW_RIGHT_WORDS + 1)
        textLda  = " ".join(chunkTokensOrig[startLda : endLda])
        ldaTopics = getLDATopicsOf(textLda)
        for (topicIndex, probability) in ldaTopics:
            feature = 'lda%d=%d'.format(topicIndex, int(round(probability, 1)))
            features.append(feature)
