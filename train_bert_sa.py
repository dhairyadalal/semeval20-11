# ## BERT Self-Attention Model 

# %%
import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from layers import SelfAttention

from sklearn.metrics import accuracy_score, f1_score

from transformers import BertModel, BertTokenizer, DistilBertTokenizer,    DistilBertModel

from typing import List

def get_lens(batch: torch.tensor) -> torch.tensor:
    batch = batch.detach().cpu()
    lens = [len(np.where(row>0)[0]) for row in batch]
    return torch.tensor(lens)

class BertAttentionClassifier(pl.LightningModule):
    
    def __init__(self, 
                 num_classes: int):
        super(BertAttentionClassifier, self).__init__()
    
        self.bert = BertModel.from_pretrained('bb_lm_ft/')
        self.num_classes = num_classes
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.self_attention = SelfAttention(256,batch_first=True, non_linearity="tanh")
        self.out = nn.Linear(256, num_classes)
                    
    def forward(self, 
               input_ids: torch.tensor,
               sent_lengths: List[int]):
        h, attn = self.bert(input_ids=input_ids)
        linear1 = torch.nn.functional.relu(self.linear1(h))        
        attention, _ = self.self_attention(linear1, sent_lengths)
        logits = self.out(attention)
        return logits, attn
    
    def training_step(self, batch, batch_idx):
        # batch
        input_ids, labels = batch
        sent_lengths = get_lens(input_ids)
        
        # predict
        y_hat, attn = self.forward(input_ids, sent_lengths)
        
        # loss 
        loss = F.cross_entropy(y_hat, labels)

        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        sent_lengths = get_lens(input_ids)
        
        y_hat, attn = self.forward(input_ids, sent_lengths)
        
        loss = F.cross_entropy(y_hat, labels)
        
        a, y_hat = torch.max(y_hat, dim=1)
        y_hat = y_hat.cpu()
        labels = labels.cpu()

        val_acc = accuracy_score(labels, y_hat)
        val_acc = torch.tensor(val_acc)
        
        val_f1 = f1_score(labels, y_hat, average='micro')
        val_f1 = torch.tensor(val_f1)

        return {'val_loss': loss, 'val_acc': val_acc, 'val_f1': val_f1}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_val_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc, 'avg_val_f1': avg_val_f1}
        
        return {'avg_val_loss': avg_loss, 'avg_val_f1':avg_val_f1 ,'progress_bar': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        input_ids, labels = batch
        sent_lengths = get_lens(input_ids)
        
        y_hat, attn = self.forward(input_ids, sent_lengths)
        
        loss = F.cross_entropy(y_hat, labels)
        
        a, y_hat = torch.max(y_hat, dim=1)
        y_hat = y_hat.cpu()
        labels = labels.cpu()

        val_acc = accuracy_score(labels, y_hat)
        val_acc = torch.tensor(val_acc)
        
        val_f1 = f1_score(labels, y_hat, average='micro')
        val_f1 = torch.tensor(val_f1)

        return {'val_loss': loss, 'val_acc': val_acc, 'val_f1': val_f1}
    
    def test_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_val_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc, 'avg_val_f1': avg_val_f1}
        
        return {'avg_val_loss': avg_loss, 'avg_val_f1':avg_val_f1 ,'progress_bar': tensorboard_logs}
            
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], 
                                lr=2e-05, eps=1e-08)

    @pl.data_loader
    def train_dataloader(self):
        return train_dataloader_
    
    @pl.data_loader
    def val_dataloader(self):
        return val_dataloader_
    
    @pl.data_loader
    def test_dataloader(self):
        return test_dataloader_
    

# %% [markdown]
# ## Train Model

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from transformers import BertTokenizer, DistilBertTokenizer
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

dat = pd.read_csv("data/task_2_data.csv")
le = LabelEncoder()

train = dat[dat["source"]=="train"]
dev = dat[dat["source"]!="train"]

le = le.fit(train["label"])
train["encoded_label"] = le.fit_transform(train["label"]) 
train["num_words"] = train["text"].apply(lambda x: len(x.split()))

random_seed = 1956
tokenizer = BertTokenizer.from_pretrained('bb_lm_ft/')

train, val = train_test_split(train, test_size=.15,
                              stratify=train["encoded_label"],
                              random_state=random_seed)


# %%
BATCH_SIZE = 24

X_train = [torch.tensor(tokenizer.encode(text)) for text in train["text"]]
X_train = pad_sequence(X_train, batch_first=True, padding_value=0)
y_train = torch.tensor(train["encoded_label"].tolist())

X_val = [torch.tensor(tokenizer.encode(text)) for text in val["text"]]
X_val = pad_sequence(X_val, batch_first=True, padding_value=0)
y_val = torch.tensor(val["encoded_label"].tolist())

ros = RandomOverSampler(random_state=random_seed)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

X_train_resampled = torch.tensor(X_train_resampled)
y_train_resampled = torch.tensor(y_train_resampled)


# %%
train_dataset = TensorDataset(X_train_resampled, y_train_resampled)
train_dataloader_ = DataLoader(train_dataset,
                               sampler=RandomSampler(train_dataset),
                               batch_size=BATCH_SIZE)

val_dataset = TensorDataset(X_val, y_val)
val_dataloader_ = DataLoader(val_dataset,
                             sampler=RandomSampler(val_dataset),
                             batch_size=BATCH_SIZE)

test_dataset = TensorDataset(X_val, y_val)
test_dataloader_ = DataLoader(test_dataset,
                             sampler=RandomSampler(test_dataset),
                             batch_size=BATCH_SIZE)


# %%
for batch in train_dataloader_:
  print(batch[0].shape, batch[1].shape)
  break


# %%
model = BertAttentionClassifier(num_classes=14)

trainer = pl.Trainer(gpus=1, 
                     default_save_path=".pl_bert_sa_logs/")
trainer.fit(model)

print("finished training")
print("savin model")
torch.save(model.state_dict(), "bb_sa_lm.pt")

