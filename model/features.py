class StartsWithUppercaseFeature(object):
    def __init__(self):
        pass
    
    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("swu=%d" % (int(word[token].istitle())))
        return result

class TokenLengthFeature(object):
    def __init__(self, max_length=30):
        self.max_length = max_length

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("l=%d" % (min(len(word), self.max_length)))
        return result

class ContainsDigitsFeature(object):    
    def __init__(self):
        self.regexpContainsDigits = re.compile(r'[0-9]+')

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("cD=%d" % (int(self.regexpContainsDigits.search(word) is not None)))
        return result
    
class ContainsPunctuationFeature(object):
    def __init__(self):
        self.regexpContainsPunctuation = re.compile(r'[\.\,\:\;\(\)\[\]\?\!]+')

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("cP=%d" % (int(self.regexpContainsPunctuation.search(word) is not None)))
        return result

class OnlyDigitsFeature(object):
    def __init__(self):
        self.regexpContainsOnlyDigits = re.compile(r'^[0-9]+$')

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("oD=%d" % (int(self.regexpContainsOnlyDigits.search(word) is not None)))
        return result

class OnlyPunctuationFeature(object):
    def __init__(self):
        self.regexpContainsOnlyPunctuation = re.compile(r'^[\.\,\:\;\(\)\[\]\?\!]+$')

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("oP=%d" % (int(self.regexpContainsOnlyPunctuation.search(word) is not None)))
        return result

class W2VClusterFeature(object):
    def __init__(self, w2v_clusters):
        self.w2v_clusters = w2v_clusters

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("w2v=%d" % (self.token_to_cluster(token)))
        return result
    
    def token_to_cluster(token):
        return self.w2v_clusters.get_cluster_of(token, -1)

class BrownClusterFeature(object):
    def __init__(self, brown_clusters):
        self.brown_clusters = brown_clusters

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("bc=%d" % (self.token_to_cluster(token)))
        return result
    
    def token_to_cluster(token):
        return self.brown_clusters.get_cluster_of(token, -1)

class BrownClusterBitsFeature(object):
    def __init__(self, brown_clusters):
        self.brown_clusters = brown_clusters

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("bcb=%s" % (self.token_to_bitchain(token)[0:7]))
        return result
    
    def token_to_bitchain(token):
        return self.brown_clusters.get_bitchain_of(token, "")

class GazetteerFeature(object):
    def __init__(self, gazetteer):
        self.gazetteer = gazetteer
    
    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("g=%d" % (int(self.is_in_gazetteer(token))))
        return result
    
    def is_in_gazetteer(token):
        return self.gazetteer.is_in_gazetteer(token)

class WordPatternFeature(object):
    def __init__(self):
        self.max_length = 15
        self.max_length_char = "~"
        
        self.normalization = [
            (r"[A-ZÄÖÜ]", "A"),
            (r"[a-zäöüß]", "a"),
            (r"[0-9]", "9"),
            (r"[\.\!\?\,\;]", "."),
            (r"[\(\)\[\]\{\}]", "("),
            (r"[^Aa9\.\(]", "#")
        ]
        
        # note: we do not map numers to 9+, e.g. years will still be 9999
        self.mappings = [
            (r"[A]{2,}", "A+"),
            (r"[a]{2,}", "a+"),
            (r"[\.]{2,}", ".+"),
            (r"[\(]{2,}", "(+"),
            (r"[#]{2,}", "#+")
        ]

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("wp=%s" % (self.token_to_wordpattern(token)))
        return result
    
    def token_to_wordpattern(token):
        normalized = token
        for from_regex, to_str in self.normalization:
            normalized = re.sub(from_regex, to_str, normalized)
        
        wp = normalized
        for from_regex, to_str in self.mappings:
            wp = re.sub(from_regex, to_str, wp)
        
        if len(wp) > self.max_length:
            wordPattern = wp[0:max_length] + self.max_length_char
        
        return wordPattern

class UnigramRankFeature(object):
    def __init__(self, unigrams):
        self.unigrams = unigrams

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            result.append("ng1=%d" % (self.token_to_rank(token)))
        return result
    
    def token_to_rank(token):
        return self.unigrams.get_rank_of(token, -1)

class PrefixFeature(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            prefix = re.sub(r'[^a-zA-ZäöüÄÖÜß\.\,\!\?]', '#', word[0:3])
            result.append("pf=%s" % (prefix))
        return result

class SuffixFeature(object):
    def __init__(self):
        pass

    def convert_sentence(sentence):
        result = []
        for token in sentence.tokens:
            suffix = re.sub(r'[^a-zA-ZäöüÄÖÜß\.\,\!\?]', '#', word[-3:])
            result.append("sf=%s" % (prefix))
        return result

class POSTagFeature(object):
    def __init__(self, pos_tagger):
        self.pos_tagger = pos_tagger

    def convert_sentence(sentence):
        pos_tags = self.stanford_pos_tag(sentence)
        result = []
        for i, token in enumerate(sentence.tokens):
            result.append("pos=%s" % (pos_tags[i]))
        return result
    
    def stanford_pos_tag(sentence):
        return self.tag(sentence)

class LDATopicFeature(object):
    def __init__(self, lda_wrapper, window_left_size, window_right_size, prob_threshold=0.5):
        self.lda_wrapper = lda_wrapper
        self.window_left_size = window_left_size
        self.window_right_size = window_right_size
        self.prob_threshold = prob_threshold

    def convert_sentence(sentence):
        result = []
        for i, token in enumerate(sentence.tokens):
            window_start = max(0, i - self.window_left_size)
            window_end = min(len(sentence), i + self.window_right_size + 1)
            window_tokens = sentence.tokens[window_start:window_end]
            text = " ".join(window_tokens)
            topics = self.get_topics_of(text)
            for (topic_idx, prob) in topics:
                if prob > self.prob_threshold:
                    result.append("lda_%d=%s" % (topic_idx, "1"))
        return result
    
    def get_topics_of(text):
        return self.lda_wrapper.get_topics_of(text)
