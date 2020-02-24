#%%
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import numpy as np

train_dat = pd.read_csv("../task1_data.csv")
dev_dat = pd.read_csv("../task1_dev.csv")
test_dat = pd.read_csv("../task1_test.csv")

# %%
random_seed = 1988

train, val = train_test_split(train_dat, 
                              test_size=.15, 
                              random_state=random_seed,
                              stratify=train_dat["span_flag"])

ros = RandomOverSampler(random_state=random_seed)

i = np.array(train["span_flag"].tolist()).reshape(-1,1)
ii = np.array(train["span_flag"].tolist())

_, __ = ros.fit_resample(i, ii)

sample_idx = ros.sample_indices_
X_train_rs = [train.iloc[idx]["text"] for idx in sample_idx]
y_train_rs = [train.iloc[idx]["seq_annotation"] for idx in sample_idx]

#%%
# Write out train dataset 
from typing import List
def write_file(vals: List[str], filename:str):
    with open(filename, "w") as f:
        for val in  vals:
            f.write(val.strip() + "\n")

write_file(X_train_rs, "train.src")
write_file(val["text"], "val.src")
write_file(dev_dat["text"], "dev.src") 
write_file(test_dat["text"], "test.src") 
        
# %%
with open("val.trg", "w") as f:
    for i, v in val.iterrows():
        annot = v["seq_annotation"]
        if annot != annot:
            f.write("[NONE]\n")
        else:
            f.write(annot.strip() + "\n")


# %%
with open("train.trg", "w") as f:
    for annot in y_train_rs:
        if annot != annot:
            f.write("[NONE]\n")
        else:
            f.write(annot.strip() + "\n")


# %%
