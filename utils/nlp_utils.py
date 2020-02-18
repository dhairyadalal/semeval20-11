from string import punctuation
from typing import List

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import swifter

lemmatizer = WordNetLemmatizer()

def tokenize_and_normalize_string(text: str) -> List[str]:
    """Lowercase, lematize and tokenize string. We convert punctuations into
    the PUNCT token. Returns a list of tokens"""
    
    text = text.lower()
    toks = [ lemmatizer.lemmatize(tok if tok not in punctuation else "PUNCT") 
             for tok in word_tokenize(text)]
    return toks

def numericalize_tokens(tokens: List[str], vocab_map: dict) -> List[int]:
    """ Given a list of tokens and vocab map, convert tokens to 
        numerical ids """
    try:
        return [vocab_map[tok] if tok in vocab_map else vocab_map["UNK"] 
                for tok in tokens]
    except:
        print("broke numer", tokens)