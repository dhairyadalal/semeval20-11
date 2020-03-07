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
dat = dat[dat["span_flag"] == 1]

tag_to_idx = {"O": 0, "B-I": 1, "I-I": 2, "X": 3}
idx_to_tag  = ["O", "B-I", "I-I", "X"]

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

MAX_LEN = 180
BATCH_SIZE = 20

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

X_train_rs = torch.tensor(X_train)
X_val = torch.tensor(X_val)

tr_masks_rs = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

y_train_rs = torch.tensor(y_train)
y_val = torch.tensor(y_val)


train_data = TensorDataset(X_train_rs, tr_masks_rs, y_train_rs)
train_dataloader_ = DataLoader(train_data,
                               sampler=RandomSampler(train_data),
                               batch_size=BATCH_SIZE)

val_data = TensorDataset(X_val, val_masks, y_val)
val_dataloader_ = DataLoader(val_data, 
                             batch_size=BATCH_SIZE)

#%%
import pytorch_lightning as pl
import torch.nn as nn
from transformers import BertForSequenceClassification
from layers import self_attention, CRF
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

def flat_accuracy(preds, labels):
    return np.sum(preds == labels) / len(labels)

# %%
class BertCRFClassifier(pl.LightningModule):
    
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
        # pooled_logits = torch.mean(torch.stack(bert_logits), dim=0)        
        return crf_out, bert_out
    
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
        param_optimizer = list(self.parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
             "weight_decay": 0.0}
        ]
        optimizer = Adam(optimizer_grouped_parameters, lr=2e-5)
        # optimizer = Adam(optimizer_grouped_parameters, lr=5e-5)
        
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

model.load_state_dict(torch.load("saved_models/bert_large_span_clf_solo.pt"))
trainer = pl.Trainer(gpus=1, 
                     default_save_path="logs/task1_bertcrf_large_solo_logs/",
                     max_epochs=40,
                     accumulate_grad_batches=50,
                     gradient_clip_val=5.0)
trainer.fit(model)
torch.save(model.state_dict(), "saved_models/bert_large_span_clf_solo_lon.pt")

