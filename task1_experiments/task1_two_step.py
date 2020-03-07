#%%
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification,\
     BertForTokenClassification, BertTokenizer
from layers import self_attention, CRF
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

class BertCRFClassifier(nn.Module):
    
    def __init__(self, 
                 num_classes: int,
                 bert_weights: str):
        super(BertCRFClassifier, self).__init__()
        
        self.bert = BertForTokenClassification.from_pretrained(bert_weights,
                                                               num_labels=4,
                                                      output_hidden_states=True)
        self.crf = CRF(num_tags=num_classes)
        
    def forward(self, 
                input_ids: torch.tensor,
                attention_mask: torch.tensor=None):
        bert_out, _ = self.bert(input_ids, attention_mask=attention_mask)
        crf_out = self.crf(bert_out) 
        return crf_out, bert_out

class BertBinaryClassifier(nn.Module):
    
    def __init__(self, 
                 bert_weights: str):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_weights,
                                                                  num_labels=2)
       
    def forward(self, 
                input_ids: torch.tensor,
                attention_mask: torch.tensor=None):
        return self.bert(input_ids, attention_mask=attention_mask)
            

WEIGHT = "bert-large-uncased"
binary_model = BertBinaryClassifier(WEIGHT)

# %%
binary_model.load_state_dict(torch.load("saved_models/bert_large_binary.pt"))

# %%
span_model = BertCRFClassifier(num_classes=4, bert_weights="bert-base-uncased")
span_model.load_state_dict(torch.load("saved_models/bert_large_span_clf_solo.pt"))

# %%
from typing import List
def extract_pred_spans(prediction: List[int]) -> List[tuple]:
    spans = []
    start_idx = -1
    end_idx = -1

    for i, val in enumerate(prediction):
        if val in [1,2] and start_idx == -1:
            start_idx = i 
        if val == 0 and start_idx != -1 and end_idx == -1:
            end_idx = i-1
        if start_idx != -1 and end_idx != -1:
            spans.append((start_idx, end_idx))
            start_idx = -1
            end_idx = -1
    return spans

def normalize_text(text: str) -> str:
    
    text = text.replace("#","").replace(' ’ ', '’').replace(" - ","-").replace(" @ "," @")
    text = text.replace(" * ", "*") 
    return text
    
#%%

from utils import get_article
import re 
from fuzzysearch import find_near_matches
from tqdm import tqdm
import pandas as pd

dev = pd.read_csv("data/task1_dev.csv")
test = pd.read_csv("data/task1_test.csv")


device1 = "cuda:0"
device2 = "cuda:1"

binary_model = binary_model.to(device1)
span_model = span_model.to(device2)

binary_model.eval()
span_model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

pred_rows = []
for i,v in dev.iterrows():
        line = v["text"]
        toks = tokenizer.tokenize(line)
        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(toks)])
        input_ids2 = torch.tensor([tokenizer.convert_tokens_to_ids(toks)])
        
        article = get_article(v["article_path"]).lower()
                
        with torch.no_grad():
                input_ids = input_ids.to(device1)
                binary_pred = binary_model(input_ids)[0]

                if torch.argmax(binary_pred) == 1:
                        input_ids = input_ids.to(device2)
                        preds, log = span_model(input_ids)
                else:
                        continue
        preds = preds[0].cpu().tolist()
        spans = extract_pred_spans(preds)
        
        if len(spans) == 0:
                continue
        
        for span in spans:
                if span[0] == span[1]:
                        pred_text = tokenizer.decode([input_ids[0][span[0]]])
                else:
                        pred_text = tokenizer.decode(input_ids[0][span[0]: span[1]])
                        pred_text = normalize_text(pred_text)
                
                if len(pred_text) < 4:
                        continue
                max_l_dists = [0,1,2,3,4,5,6,7,8,9]
                for dist in max_l_dists:
                        match = find_near_matches(pred_text, article, max_l_dist=dist)
                        if len(match) > 0:
                                pred_rows.append({"article_id": v["article_id"],
                                                "pred_text": pred_text,
                                                "span_start": match[0].start,
                                                "span_end": match[0].end})
                        break  

#%%
with open("submissions/task1/task1_bert_2step_1_dev.txt", "w") as f:
    for row in pred_rows:
        f.write(f"{row['article_id']}\t{row['span_start']}\t{row['span_end']}\n")


#%%
# def get_predictions(df: pd.DataFrame) -> List[dict]:
#     pred_rows = []
#     for i,v in tqdm(df.iterrows(), total= len(df)):
#         line = v["text"]
#         toks = tokenizer.tokenize(v["text"])
#         ids  = torch.tensor([tokenizer.convert_tokens_to_ids(toks)]) 
#         article = get_article(v["article_path"]).lower()
        
#         with torch.no_grad():
#             ids = ids.to("cuda:0")
#             preds = model(ids, attention_mask=None, labels=None)
#             preds = torch.argmax(preds[0], dim=2)
#         preds = preds.tolist()[0]
#         spans = extract_pred_spans(preds)
        
#         for span in spans:
#             if span[0] == span[1]:
#                 pred_text = tokenizer.decode([ids[0][span[0]]])
#             else:
#                 pred_text = tokenizer.decode(ids[0][span[0]: span[1]])
#             pred_text = normalize_text(pred_text)
            
#             if len(pred_text) < 4:
#                 continue
            
#             # Gradually increase fuzzy match dist to prevent bad match for short spans
#             max_l_dists = [0,1,2,3,4,5,6,7,8,9]
#             for dist in max_l_dists:
#                 match = find_near_matches(pred_text, article, max_l_dist=dist)
#                 if len(match) > 0:
#                     pred_rows.append({"article_id": v["article_id"],
#                                 "pred_text": pred_text,
#                                 "span_start": match[0].start,
#                                 "span_end": match[0].end})
#                     break        
                
#     return pred_rows

dev = pd.read_csv("data/task1_dev.csv")
test = pd.read_csv("data/task1_test.csv")

print("generating dev preds")
dev_preds = get_predictions(dev)

print("generating test preds")
test_preds = get_predictions(test)

with open("submissions/task1/task1_bert_dev.txt", "w") as f:
    for row in dev_preds:
        f.write(f"{row['article_id']}\t{row['span_start']}\t{row['span_end']}\n")

with open("submissions/task1/task1_bert_test.txt", "w") as f:
    for row in test_preds:
        f.write(f"{row['article_id']}\t{row['span_start']}\t{row['span_end']}\n")
