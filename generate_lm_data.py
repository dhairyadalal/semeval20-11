#%%
from os import listdir
from os.path import isfile, join

mypath = "data/train-articles/"
train_files = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]

mypath = "data/dev-articles/"
dev_files = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]

mypath = "data/test-articles/"
test_files = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]

# %%
import pandas as pd

t = pd.DataFrame({"file":train_files})
t["src"] = "train"

d = pd.DataFrame({"file":dev_files})
d["src"] = "dev"
tt = pd.DataFrame({"file":test_files})
tt["src"] = "test"


df = pd.concat([t,d])

# %%
from sklearn.model_selection import train_test_split

random_state = 1024
train, val = train_test_split(df, test_size=.15, random_state=random_state,
                              stratify = df["src"])

# %%
from utils import get_article

with open("data/lm_train_v2.txt", "w") as f:
    for x in train["file"]:
        f.write(get_article(x))


with open("data/lm_val.txt_v2", "w") as f:
    for x in val["file"]:
        f.write(get_article(x))

# %%