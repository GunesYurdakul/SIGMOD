import math


import re
import math

#USAGE CharNgrams('onur is the best coder',3)*CharNgrams('is the onur best coder',3)
#Text for first parameter, n for the second 
import re
import math

class Ngrams(object):
    """Compare strings using an n-grams model and cosine similarity.
    This class uses words as tokens. See module docs.
    >>> sorted(Ngrams('''Compare strings using an n-grams model and cosine similarity. This class uses words as tokens. See module docs.''').d.items())
    [('an ngrams model', 0.23570226039551587), ('and cosine similarity', 0.23570226039551587), ('as tokens see', 0.23570226039551587), ('class uses words', 0.23570226039551587), ('compare strings using', 0.23570226039551587), ('cosine similarity this', 0.23570226039551587), ('docs', 0.23570226039551587), ('model and cosine', 0.23570226039551587), ('module docs', 0.23570226039551587), ('ngrams model and', 0.23570226039551587), ('see module docs', 0.23570226039551587), ('similarity this class', 0.23570226039551587), ('strings using an', 0.23570226039551587), ('this class uses', 0.23570226039551587), ('tokens see module', 0.23570226039551587), ('uses words as', 0.23570226039551587), ('using an ngrams', 0.23570226039551587), ('words as tokens', 0.23570226039551587)]
    """
    ngram_joiner = " "

    class WrongN(Exception):
        """Error to raise when two ngrams of different n's are being
        compared.
        """
        pass

    def __init__(self, text, n=3):
        self.n = n
        self.text = text
        self.d = self.text_to_ngrams(self.text, self.n)

    def __getitem__(self, word):
        return self.d[word]

    def __contains__(self, word):
        return word in self.d

    def __iter__(self):
        return iter(self.d)

    def __mul__(self, other):
        """Returns the similarity between self and other as a float in
        (0;1).
        """
        if self.n != other.n:
            raise self.WrongN
        if self.text == other.text:
            return 1.0
        return sum(self[k]*other[k] for k in self if k in other)

    def __repr__(self):
        return "Ngrams(%r, %r)" % (self.text, self.n)

    def __str__(self):
        return self.text

    def tokenize(self, text):
        """Return a sequence of tokens from which the ngrams should be constructed.
        This shouldn't be a generator, because its length will be
        needed.
        """

        return re.compile(u'[^\w\n ]|\xe2', re.UNICODE).sub('', text).lower().split()

    def normalize(self, text):
        """This method is run on the text right before tokenization"""
        try:
            return text.lower()
        except AttributeError:
            # text is not a string?
            raise TypeError(text)
    
    def make_ngrams(self, text):
        """
        # -*- coding: utf-8 -*-
        Return an iterator of tokens of which the n-grams will
        consist. You can overwrite this method in subclasses.
        >>> list(Ngrams('').make_ngrams(chr(10).join([u"This work 'as-is' we provide.",\
        u'No warranty, express or implied.', \
        u"We've done our best,", \
        u'to debug and test.',\
        u'Liability for damages denied.'])))[:5]
        [u'this work asis', u'work asis we', u'asis we provide', u'we provide no', u'provide no warranty']
        """
        text = self.normalize(text)
        tokens = self.tokenize(text)
        return (self.ngram_joiner.join(tokens[i:i+self.n]) for i in range(len(tokens)))

    def text_to_ngrams(self, text, n=3):
        d = {}
        for ngram in self.make_ngrams(text):
            try: d[ngram] += 1
            except KeyError: d[ngram] = 1

        norm = math.sqrt(sum(x**2 for x in d.values()))
        for k, v in d.items():
            d[k] = v/norm
        return d

class CharNgrams(Ngrams):

    """Ngrams comparison using single characters as tokens.
    >>> CharNgrams("ala ma kota")*CharNgrams("ala ma kota")
    1.0
    >>> round(CharNgrams("This Makes No Difference") * CharNgrams("this makes no difference"), 4)
    1.0
    >>> CharNgrams("Warszawska")*CharNgrams("Warszawa") > CharNgrams("Warszawa")*CharNgrams("Wawa")
    True
    """
    ngram_joiner = ''
    def tokenize(self, st):
        """
        >>> ''.join(CharNgrams('').tokenize('ala ma kota!'))
        'alamakota'
        """
        return [c for c in st if c.isalnum()]

class CharNgramSpaces(CharNgrams):
    '''Like CharNgrams, except it keeps whitespace as one space in
    the process of tokenization. This should be useful for analyzing
    texts longer than words, where places at which word boundaries
    occur may be important.'''
    def tokenize(self, st):
        return super(CharNgramSpaces, self).tokenize(re.sub(r'\s+', ' ', st))







def tokenize_words_nospace(st):
    '''Like CharNgrams, except it keeps whitespace as one space in
    the process of tokenization. This should be useful for analyzing
    texts longer than words, where places at which word boundaries
    occur may be important.'''

    return tokenize_chars(re.sub(r'\s+', ' ', st))

def tokenize_words(text):
    """Return a sequence of tokens from which the ngrams should be constructed.
    This shouldn't be a generator, because its length will be
    needed.
    """
    return re.compile(u'[^\w\n ]|\xe2', re.UNICODE).sub('', text).lower().split()


def tokenize_chars(st):
    """Return a sequence of tokens from which the ngrams should be constructed.
    This shouldn't be a generator, because its length will be
    needed.
    """
    return [c for c in st if c.isalnum()]

def make_ngrams(text,n,typeof):
   
    ngram_joiner = ''
    #text = self.normalize(text)
    if (typeof=="word"):
        tokens = tokenize_words(text)
    if (typeof == "chars"):
        tokens = tokenize_chars(text)
    if (typeof == "spaces"):
        tokens = tokenize_words_nospace(text)
        
    return (ngram_joiner.join(tokens[i:i+n]) for i in range(len(tokens)))


#Produce ngrams of a given text. type of can be : word, chars or spaces 
def text_to_ngrams(text, n,typeof):
    d = {}
    for ngram in make_ngrams(text,n,typeof):
        try: d[ngram] += 1
        except KeyError: d[ngram] = 1

    norm = math.sqrt(sum(x**2 for x in d.values()))
    for k, v in d.items():
        d[k] = v/norm
    return d



#Similarity functions

def min_feature_size(query_size, alpha):
    return int(math.ceil(alpha * alpha * query_size))

def max_feature_size(query_size, alpha):
    return int(math.floor(query_size * 1.0 / (alpha * alpha)))

def minimum_common_feature_count(query_size, y_size, alpha):
    return int(math.ceil(alpha * math.sqrt(query_size * y_size)))

def cosine_sim(X, Y):
    return len(set(X) & set(Y)) * 1.0 / math.sqrt(len(set(X)) * len(set(Y)))


def min_feature_size(self, query_size, alpha):
    return int(math.ceil(alpha * query_size))

def max_feature_size(query_size, alpha):
    return int(math.floor(query_size / alpha))

def minimum_common_feature_count(query_size, y_size, alpha):
    return int(math.ceil(alpha * (query_size + y_size) * 1.0 / (1 + alpha)))

def jaccard_sim(X, Y):
    return len(set(X) & set(Y)) * 1.0 / len(set(X) | set(Y))



def min_feature_size(query_size, alpha):
    return int(math.ceil(alpha * 1.0 / (2 - alpha) * query_size))

def max_feature_size(query_size, alpha):
    return int(math.floor((2 - alpha) * query_size * 1.0 / alpha))

def minimum_common_feature_count(query_size, y_size, alpha):
    return int(math.ceil(0.5 * alpha * query_size * y_size))

def dice_sim(X, Y):
    return len(set(X) & set(Y)) * 2.0 / (len(set(X)) + len(set(Y)))