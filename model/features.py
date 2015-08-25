class StartsWithUppercase(object):
    def __init__(self):
        pass
    
    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("swu=%d" % (int(word[token].istitle())))
        return result

class TokenLength(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("l=%d" % (min(len(word), 30)))
        return result

class ContainsDigits(object):    
    def __init__(self):
        self.regexpContainsDigits = 

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("cD=%d" % (int(self.regexpContainsDigits.search(word) is not None)))
        return result
    
class ContainsPunctuation(object):
    def __init__(self):
        self.regexpContainsPunctuation = 

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("cP=%d" % (int(self.regexpContainsPunctuation.search(word) is not None)))
        return result

class OnlyDigits(object):
    def __init__(self):
        self.regexpContainsOnlyDigits = 

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("oD=%d" % (int(self.regexpContainsOnlyDigits.search(word) is not None)))
        return result

class OnlyPunctuation(object):
    def __init__(self):
        self.regexpContainsOnlyPunctuation = 

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("oP=%d" % (int(self.regexpContainsOnlyPunctuation.search(word) is not None)))
        return result

class W2VCluster(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("w2v=%d" % (self.token_to_cluster(token)))
        return result
    
    def token_to_cluster(token):
        pass

class BrownCluster(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("bc=%d" % (self.token_to_cluster(token)))
        return result
    
    def token_to_cluster(token):
        pass

class BrownClusterBits(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("bcb=%s" % (self.token_to_bitchain(token)[0:7]))
        return result
    
    def token_to_bitchain(token):
        pass

class Gazzetteer(object):
    def __init__(self):
        pass
    
    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("g=%d" % (int(self.is_in_gazetteer(token))))
        return result
    
    def is_in_gazetteer(token):
        pass

class WordPattern(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("wp=%s" % (self.token_to_wordpattern(token)))
        return result

class UnigramRank(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("ng1=%d" % (self.token_to_rank(token)))
        return result
    
    def token_to_rank(token):
        pass

class Prefix(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            prefix = re.sub(r'[^a-zA-ZäöüÄÖÜß\.\,\!\?]', '#', word[0:3])
            result.append("pf=%s" % (prefix))
        return result

class Suffix(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            suffix = re.sub(r'[^a-zA-ZäöüÄÖÜß\.\,\!\?]', '#', word[-3:])
            result.append("sf=%s" % (prefix))
        return result

class POSTag(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        pos = self.stanford_pos_tag(sentence)
        result = []
        for i, token in enumerate(sentence.tokens):
            result.append("pos=%s" % (pos[i]))
        return result
    
    def stanford_pos_tag(sentence):
        pass

class LDATopic(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        result = []
        for i, token in enumerate(sentence.tokens):
            start = max(0, i - LDA_WINDOW_LEFT_WORDS)
            end = min(len(sentence), i + LDA_WINDOW_RIGHT_WORDS + 1)
            text = " ".join(sentence[start:end])
            topics = self.get_topics_of(text)
            for (topic_idx, prob) in topics:
                if prob > 0.75:
                    result.append("lda_%d=%s" % (topic_idx, "1"))
        return result
    
    def get_topics_of(sentence):
        pass
