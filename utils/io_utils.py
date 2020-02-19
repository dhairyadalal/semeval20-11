import re
from typing import Tuple, List
from collections import namedtuple

T1Span = namedtuple("T1Span", ['start', 'end', 'text'])
T2Span = namedtuple("T2Span", ['start', 'end', 'text', 'label'])

def get_article(file: str) -> str:
    """ Read file and return raw text"""
    with open(file, "r") as f:
        lines = f.read()
    return lines

def get_gold_spans(file: str) -> List[Tuple[int,int]]:
    spans = list()
    with open(file, "r") as f:
        for line in f.readlines():
            ls = line.split('\t')
            spans.append( (int(ls[1].strip()), int(ls[2].strip())))
    return spans

def get_gold_labels(file: str) -> List[dict]:
    spans = list()
    with open(file, "r") as f:
        for line in f.readlines():
            ls = line.split('\t')
            spans.append({"start": int(ls[2].strip()),
                          "end": int(ls[3].strip()),
                          "label": ls[1].strip()})
    return spans
    
def get_span(text:str, article: str) -> Tuple[int, int]:    
    search = re.search(re.escape(text), article)
    if search is None:
        print("Bad search", text)
        return (-1,-1)
    return search.span()

def get_span_text(start: int, end: int, article: str) -> str:
    return article[start: end]

def get_task1_file(article_id: int) -> str:
    prefix = "data/train-labels-task1-span-identification/"
    return prefix+"article"+article_id+".task1-SI.labels"

def get_task2_file(article_id: int) -> str:
    prefix = "data/train-labels-task2-technique-classification/"
    return prefix+"article"+article_id+".task2-TC.labels"

def get_article_file(article_id: int) -> str:
    return "data/train-articles/article" + str(article_id) + ".txt"

def get_dev_article_file(article_id: int) -> str:
    return "data/dev-articles/article" + str(article_id) + ".txt"