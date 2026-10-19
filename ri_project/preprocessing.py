"""Tokenização e normalização lexical compartilhadas pelos métodos lexicais."""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))

_STEMMER = PorterStemmer()


def tokenize(text):
    """Normaliza para minúsculas e retorna tokens sem stemming."""
    return re.findall(r"\b\w+\b", str(text).lower())


def preprocess_tokens(tokens):
    """Remove stopwords e aplica stemming a tokens já normalizados."""
    return [_STEMMER.stem(token) for token in tokens if token not in STOPWORDS]


def preprocess(text):
    """Tokeniza, normaliza, remove stopwords e aplica stemming Porter."""
    return preprocess_tokens(tokenize(text))
