#%%
from typing import List

from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import BertTokenizer, BertForTokenClassification, BertModel
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
import numpy as np

dat = pd.read_csv("data/task1_data.csv")

tag_to_idx = {"O": 0, "B-I": 1, "I-I": 2, "X": 3}
idx_to_tag  = ["O", "B-I", "I-I", "X"]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

MAX_LEN = 180
BATCH_SIZE = 24

def labels_to_ids(annotation: List[str]) -> List[int]:
    return [tag_to_idx[i] for i in annotation]

dat["bert_len"] = dat["bert_tokenized_text"].apply(lambda x: len(x.split()))
dat = dat.sort_values(by=["bert_len"])

input_ids = [tokenizer.convert_tokens_to_ids(text.split()) \
            for text in dat["bert_tokenized_text"]]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                          truncating="post", 
                          padding="post")

attn_mask = [[float(i>0) for i in ii] for ii in input_ids]

label_ids = [ labels_to_ids(a.split()) for a in dat["bert_annotation"]]
label_ids = pad_sequences(label_ids, maxlen=MAX_LEN, dtype="long", 
                          truncating="post", 
                          padding="post")
span_flag = dat["span_flag"]

random_state = 1988
X_train, X_val, y_train, y_val  =  train_test_split(input_ids, label_ids,
                                                    random_state=random_state,
                                                    test_size=.15,
                                                    stratify=span_flag)

tr_masks, val_masks, \
    tr_span_flag, val_span_flag = train_test_split(attn_mask, span_flag,
                                                   random_state=random_state,
                                                   test_size=.15,
                                                   stratify=span_flag)

ros = RandomOverSampler(random_state=random_state)
X_train_rs, y_strat = ros.fit_resample(X_train, tr_span_flag)
y_train_rs = [y_train[idx] for idx in ros.sample_indices_]
tr_masks_rs = [tr_masks[idx] for idx in ros.sample_indices_]

X_train_rs = torch.tensor(X_train_rs)
X_val = torch.tensor(X_val)

tr_masks_rs = torch.tensor(tr_masks_rs)
val_masks = torch.tensor(val_masks)

y_train_rs = torch.tensor(y_train_rs)
y_val = torch.tensor(y_val)


train_data = TensorDataset(X_train_rs, tr_masks_rs, y_train_rs)
train_dataloader_ = DataLoader(train_data,
                               sampler=RandomSampler(train_data),
                               batch_size=BATCH_SIZE)

val_data = TensorDataset(X_val, val_masks, y_val)
val_dataloader_ = DataLoader(val_data, 
                             batch_size=BATCH_SIZE)
from seqeval.metrics import f1_score

#%%
import pytorch_lightning as pl
import torch.nn as nn
from layers import CRF
from transformers import AdamW

class BertCRFClassifier(pl.LightningModule):
    
    def __init__(self, 
                 num_classes: int,
                 bert_weights: str,
                 dropout: float=.10):
        super(BertCRFClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_weights)
        hidden_size = self.bert.config.hidden_size
        
        self.lstm = torch.nn.LSTM(128,
                                  hidden_size=128,
                                  bidirectional=True,
                                  batch_first=True)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear1 = torch.nn.Linear(hidden_size, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128*2, num_classes)
        self.relu = torch.nn.ReLU()
        self.crf = CRF(num_tags=num_classes)
        
    def forward(self, 
                input_ids: torch.tensor,
                attention_mask: torch.tensor=None):
        bert_out, _ = self.bert(input_ids, attention_mask=attention_mask)
        lin1 = self.dropout(self.relu(self.linear1(bert_out)))
        lin2 = self.dropout(self.relu(self.linear2(lin1)))
        lstm_out, _ = self.lstm(lin2)
        linear_out = self.dropout(self.linear3(lstm_out))
        crf_out = self.crf(linear_out) 
        return crf_out, linear_out
    
    def crf_loss(self, 
                 pred_logits: torch.tensor, 
                 labels: torch.tensor) -> torch.tensor:
        return self.crf.loss(pred_logits, labels)
        
    def training_step(self, batch, batch_idx):
        
        input_ids, mask, labels = batch
        preds, logits = self.forward(input_ids, attention_mask=mask)
        loss = self.crf_loss(logits, labels)        
        
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        input_ids, mask, labels = batch
        
        preds, logits = self.forward(input_ids, attention_mask=mask)
        loss = self.crf_loss(logits, labels)
        
        labels = labels.detach().cpu().numpy().flatten()
        preds  = preds.detach().cpu().numpy().flatten()
        
        recall = recall_score(labels, preds, average="macro")
        recall = torch.tensor(recall)
        
        precision = precision_score(labels, preds, average="macro")
        precision = torch.tensor(precision)
        return {'val_loss': loss, 
                "recall": recall, 
                "precision": precision}
    
    def validation_end(self, outputs):
        
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_recall = torch.stack([x["recall"] for x in outputs]).mean()
        avg_precision = torch.stack([x["precision"] for x in outputs]).mean()
        
        tensorboard_logs = {"val_loss": avg_loss, 
                            'avg_val_recall': avg_recall,
                            'avg_val_precision': avg_precision}
        return {'avg_val_loss': avg_loss, 
                'avg_val_recall': avg_recall,
                'avg_val_precision': avg_precision,
                'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        # param_optimizer = list(self.bert.named_parameters())
        # no_decay = ['bias', 'gamma', 'beta']
        # #no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        #     'weight_decay_rate': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        #     'weight_decay_rate': 0.0}]
        
        # optimizer = Adam(optimizer_grouped_parameters, lr=2e-5)
        param_optimizer = list(self.parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
             "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=5e-5, 
                          eps=1e-8)
        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        return train_dataloader_
    
    @pl.data_loader
    def val_dataloader(self):
        return val_dataloader_

#%%

model = BertCRFClassifier(num_classes=4, 
                          bert_weights="bert-base-uncased")

trainer = pl.Trainer(gpus=1, 
                     default_save_path="logs/task1_bertcrf_base_logs/",
                     accumulate_grad_batches=50,
                     gradient_clip_val=1.0,
                     max_epochs=10)
trainer.fit(model)

# print("finished training. Saving model ....")
# torch.save(model.state_dict(), "saved_models/bert_custom.pt")

# %%
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
    

from utils import get_article
import re 
from fuzzysearch import find_near_matches
from tqdm import tqdm

def get_predictions(df: pd.DataFrame) -> List[dict]:
    model.to("cuda:0")
    model.eval()
     
    pred_rows = []
    for i,v in df.iterrows():
        line = v["text"]
        toks = tokenizer.tokenize(v["text"])
        ids  = torch.tensor([tokenizer.convert_tokens_to_ids(toks)]) 
        article = get_article(v["article_path"]).lower()
        
        with torch.no_grad():
            ids = ids.to("cuda:0")
            preds, _ = model(ids, attention_mask=None)
            
        preds = preds.tolist()[0]
        spans = extract_pred_spans(preds)
        
        for span in spans:
            if span[0] == span[1]:
                pred_text = tokenizer.decode([ids[0][span[0]]])
            else:
                pred_text = tokenizer.decode(ids[0][span[0]: span[1]])
            pred_text = normalize_text(pred_text)
            
            if len(pred_text) < 4:
                continue
            
            # Gradually increase fuzzy match dist to prevent bad match for short spans
            max_l_dists = [0,1,2,3,4,5,6,7,8,9]
            for dist in max_l_dists:
                match = find_near_matches(pred_text, article, max_l_dist=dist)
                if len(match) > 0:
                    pred_rows.append({"article_id": v["article_id"],
                                "pred_text": pred_text,
                                "span_start": match[0].start,
                                "span_end": match[0].end})
                    break    
    return pred_rows
#%%
dev = pd.read_csv("data/task1_dev.csv")
test = pd.read_csv("data/task1_test.csv")

print("generating dev preds")
dev_preds = get_predictions(dev)
pd.DataFrame(dev_preds).to_csv("submissions/task1/custom_dev.csv")

print("generating test preds")
test_preds = get_predictions(test)
pd.DataFrame(test_preds).to_csv("submissions/task1/custom_test.csv")

with open("submissions/task1/task1_bert_custom_dev.txt", "w") as f:
    for row in dev_preds:
        f.write(f"{row['article_id']}\t{row['span_start']}\t{row['span_end']}\n")

with open("submissions/task1/task1_bert_custom_test.txt", "w") as f:
    for row in test_preds:
        print("writing: ", row)
        f.write(f"{row['article_id']}\t{row['span_start']}\t{row['span_end']}\n")

