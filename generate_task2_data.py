#%%
from collections import defaultdict
from utils import get_article, get_span, get_span_text, get_article_file, \
get_task1_file, get_gold_spans, get_dev_article_file, get_article_by_id
import pandas as pd

# Extract train data
train_gold = "data/train-task2-TC.labels"
dev_file = "data/dev-task-TC-template.out"
test_file = "data/test-task-TC-template.out"

def extract_annotations(file: str, source: str) -> pd.DataFrame:

    with open(file, "r") as f:
        tc_lines = f.readlines()
    
    article_ids = [t.split("\t")[0] for t in tc_lines]
    article_ids = list(set(article_ids))

    final = []
    
    gold_spans = defaultdict(list)
    for line in tc_lines:
        ls = line.strip().split("\t")
        
        gold_spans[ls[0]].append({"label": ls[1], 
                                  "start": int(ls[2]), 
                                  "end": int(ls[3])})
    for aid in article_ids:
        article = get_article_by_id(aid, type=source) 
        
        spans = sorted(gold_spans[aid], key=lambda x: x["start"])
        
        for span in spans:
            span_text = get_span_text(span["start"], span["end"], article)
            span["context"] = "?"
            for line in article.split("\n"):
                
                line_span = get_span(line, article)
                if span["start"] >= line_span[0] and span_text in line:
                    span["context"] = line
                    break
            
            if span["context"] == "?":
                for line in list(article.split("\n"))[::-1]:
                    if span_text in line:
                        span_text["context"] = line
                        break
            if span["context"] == "?":
                span["context"] = span_text
                
            span["text"] = span_text
            span["source"] = source
            final.append(span)

    return pd.DataFrame(final)

train_annot = extract_annotations(train_gold, "train")
print("finihsed extracting train ... ")
print("movin on to dev ... ")
dev_annot = extract_annotations(dev_file, "dev")
print("finihsed extracting dev ... ")
print("movin on to test ... ")
test_annot = extract_annotations(test_file, "test")
print("finihsed")

t2_dat = pd.concat([train_annot, dev_annot, test_annot])
t2_dat.to_csv("data/task2_data.csv")
