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

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

MAX_LEN = 180
BATCH_SIZE = 16

def labels_to_ids(annotation: List[str]) -> List[int]:
    return [tag_to_idx[i] for i in annotation]

dat["bert_len"] = dat["bert_tokenized_text"].apply(lambda x: len(x.split()))
dat = dat.sort_values(by=["bert_len"])

input_ids = [tokenizer.encode(text) for text in dat["text"]]
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

#%%
import pytorch_lightning as pl
import torch.nn as nn
from transformers import BertForSequenceClassification, BertModel
from layers import SelfAttention
import torch.nn.functional as F

def flat_accuracy(preds, labels):
    return np.sum(preds == labels) / len(labels)

class BertBinaryClassifier(pl.LightningModule):
    
    def __init__(self, 
                 bert_weights: str):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_weights)
        
        # Freeze Bert Params
        for param in list(self.bert.parameters())[:-10]:
            param.requires_grad = False
        
        self.dropout = nn.Dropout(p=.10)
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.attention = SelfAttention(512, batch_first=True)
        self.clf = nn.Linear(512, 2)        
    
    def forward(self, 
                input_ids: torch.tensor,
                sent_lens: torch.tensor=None,
                attention_mask: torch.tensor=None):
        bert_out, _ = self.bert(input_ids, attention_mask=attention_mask)
        lin1 =F.relu(self.linear1(bert_out))
        attention, _ = self.attention(lin1, sent_lens)
        clf = self.clf(attention)
        return clf
            
    def training_step(self, batch, batch_idx):        
        input_ids, mask, labels = batch
        sent_lens = torch.sum(mask, dim=1).int()
        preds = self.forward(input_ids, sent_lens=sent_lens, 
                             attention_mask=mask)
        gl = (torch.sum(labels, dim=1)>1).long().to("cuda:0")
        
        loss = F.cross_entropy(preds, gl)
        
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        input_ids, mask, labels = batch
        sent_lens = torch.sum(mask, dim=1).int()
        preds = self.forward(input_ids, sent_lens=sent_lens, 
                             attention_mask=mask)
        labels = (torch.sum(labels, dim=1)>1).long().to("cuda:0")
        
        loss = F.cross_entropy(preds, labels)
        
        labels = labels.detach().cpu().numpy().flatten()
        preds  = preds.detach().cpu().numpy().flatten()
        
        acc = torch.tensor(flat_accuracy(preds, labels))
        
        tensorboard_logs = {'train_loss': loss}
       
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        
        tensorboard_logs = {"val_loss": avg_loss, 
                            'avg_val_acc': avg_acc}
        
        return {'avg_val_loss': avg_loss, 
                'avg_val_acc': avg_acc,
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
        optimizer = Adam(optimizer_grouped_parameters, lr=5e-5)
        # optimizer = Adam(optimizer_grouped_parameters, lr=5e-5)
        
        return optimizer
        
    @pl.data_loader
    def train_dataloader(self):
        return train_dataloader_
    
    @pl.data_loader
    def val_dataloader(self):
        return val_dataloader_

model = BertBinaryClassifier(bert_weights="bert-large-uncased")

trainer = pl.Trainer(gpus=1, 
                     default_save_path="logs/task1_bert_binary_clf_logs/",
                     max_epochs=10,
                     accumulate_grad_batches=100,
                     gradient_clip_val=5.0)
trainer.fit(model)

print("finished training. Saving model ....")
torch.save(model.state_dict(), "saved_models/bert_large_sa_binary.pt")

# model.load_state_dict(torch.load("saved_models/bert_large_binary.pt"))

# %%
val_data = TensorDataset(X_val, val_masks, y_val)
val_dataloader_ = DataLoader(val_data, 
                             batch_size=5)

model.eval()
model.to("cuda:0")

rows = []
all_preds = []
all_labels = []

for batch in val_dataloader_:
    ii, m, l = batch
    ii = ii.to("cuda:0")
    m = m.to("cuda:0")
    l = l.to("cuda:0")
    
    sent_lens = torch.sum(m, dim=1).int().to("cuda:0")
    
    gl = (torch.sum(l, dim=1)>1).long().to("cuda:0")
    
    preds = model(ii, attention_mask = m, sent_lens=sent_lens)
    all_preds.extend(torch.argmax(preds, dim=1).cpu().tolist())
    
    all_labels.extend(gl.cpu().tolist())
    break

from sklearn.metrics import classification_report

print(classification_report(all_labels, all_preds))


# %%
from transformers import BertModel, BertTokenizer
from layers import SelfAttention

