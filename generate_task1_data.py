#%%
from utils import get_article, get_span, get_span_text, get_article_file, \
get_task1_file, get_gold_spans, get_dev_article_file

import pickle
from nltk import word_tokenize
from unidecode import unidecode
import pandas as pd
import re 
from transformers import BertTokenizer


# Generate dev dataset for prediction
dev_ids = []
with open("data/dev-task-TC-template.out") as f:
    for line in f.readlines():
        l = line.split("\t")
        dev_ids.append(l[0].strip())
        
dev_ids = list(set(dev_ids))
dev_articles_map = {}

dev_dat = []
for i in dev_ids:
    file = get_dev_article_file(i)
    article_text = get_article(file)
    for line in article_text.split("\n"):
        if line == '':
            continue
        else:
            dev_dat.append({"dev_id": i, "text": line})
    dev_articles_map[i] = article_text

dev_df = pd.DataFrame(dev_dat)
dev_df.to_csv("data/task1_dev.csv")

# # 1. Generate train labels
with open("data/train-task1-SI.labels", "r") as f:
    train_lines = f.readlines()

article_ids = [t.split("\t")[0] for t in train_lines]
article_ids = list(set(article_ids))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

data = []
for idx, i in enumerate(article_ids):
    article_id = i
    article_file = get_article_file(i)
    article = get_article(article_file)
    
    t1_spans = get_gold_spans(get_task1_file(i)) 
    t1_spans = sorted(t1_spans, key= lambda x: x[0])
    
    article_spans = []
    for line in article.split('\n'):
        if line in article and len(line) > 1:
            article_spans.append((get_span(line, article), line))
    
    article_spans = sorted(article_spans, key= lambda k: k[0][0])
    t1_idx = 0
    t1_len = len(t1_spans)

    for pair in article_spans:
        ats = pair[0]
        line = pair[1]
        bert_line  = pair[1]
        t1s = t1_spans[t1_idx]

        if t1s[0] in range(ats[0], ats[1]):
            spans = []
            while t1s[0] in range(ats[0], ats[1]):
                span_text = get_span_text(t1s[0], t1s[1], article)   
                spans.append(span_text.replace("\n", " ").strip())
                if t1_idx + 1 < t1_len:
                    t1_idx += 1
                    t1s = t1_spans[t1_idx]
                else:
                    break
            
            span_seq_annotation = " [SEP] ".join(spans)
            for s in spans:
                annot_toks = word_tokenize(s)
                bert_toks = tokenizer.tokenize(s)
                if len(annot_toks) == 1:
                    annot = "B-I"
                    bert_annot = "B-I"
                else:
                    annot = ["B-I"]
                    annot.extend(["I-I"] * (len(annot_toks) - 1))
                    annot = " ".join(annot)
                    
                    bert_annot = []
                    for i, tok in enumerate(bert_toks):
                        if "##" in tok:
                            bert_annot.append("X")
                        elif i == 0:
                            bert_annot.append("B-I")
                        else:
                            bert_annot.append("I-I")
                    bert_annot = " ".join(bert_annot)                
                     
                line = line.replace(s, annot)
                
            line_toks = word_tokenize(line)
            line_toks = ["O" if t not in ['B-I','I-I'] else t for t in line_toks] 
            line_annot = " ".join(line_toks)          
            
            bert_line = bert_line.replace(s, bert_annot)
            bert_line_toks = tokenizer.tokenize(bert_line)
            bert_line_toks = ["O" if t not in ["B-I", "I-I", "X"] else t for t in bert_line_toks]
            bert_seq = " ".join(bert_line_toks)
            
            data.append({"article_id": article_id,
                         "text": pair[1],
                         "binary_annotation": line_annot,
                         "seq_annotation": span_seq_annotation,
                         "bert_tokenized_text": " ".join(tokenizer.tokenize(pair[1])),
                         "bert_annotation": bert_seq,
                         "span_flag": 1,
                         "span_count": len(spans)})
        else:
            # if line has no letters, skip
            if re.search('[a-zA-Z]', line) is None:
                continue
            line_annot = ["O"] * len(word_tokenize(line))
            line_annot = " ".join(line_annot)
            
            # Sentpiece annotation for Bert like models
            bert_toks = tokenizer.tokenize(line)
            bert_annot = []
            for tok in bert_toks:
                if "##" in tok:
                    bert_annot.append("X")
                else:
                    bert_annot.append("O")
            bert_annot = " ".join(bert_annot)
            
            data.append({"article_id": article_id,
                         "text": pair[1],
                         "binary_annotation": line_annot,
                         "seq_annotation": "N/A",
                         "bert_tokenized_text": " ".join(bert_toks),
                         "bert_annotation": bert_annot,
                         "span_flag": 0,
                         "span_count": 0})   

print("finished")
df = pd.DataFrame(data)
#%%
df.to_csv("data/task1_data.csv")


# %%
