#%%
from utils import get_article
from os import listdir
from os.path import isfile, join
import re 
import pandas as pd 

mypath = "data/test-articles/"
test_articles = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]

id_extractor = re.compile("\d+")

row = []
for path in test_articles:
    article_id = id_extractor.findall(path)[0]
    text = get_article(path)
    
    lines = text.split("\n")
    
    start_idx = 0
    for line in lines:
            end_idx = len(line) + start_idx + 2
            if len(line) > 1:
                row.append({"article_id": article_id,
                            "text": "[SKIP]" if len(line) < 1 else line.strip(),
                            "article_path": path,
                            "start_idx": start_idx,
                            "end_idx": end_idx})
                start_idx = end_idx + 1

df = pd.DataFrame(row)
df.to_csv("data/task1_test.csv")

# %%

mypath = "data/dev-articles/"
articles = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]

id_extractor = re.compile("\d+")

row = []
for path in articles:
    article_id = id_extractor.findall(path)[0]
    text = get_article(path)
    
    lines = text.split("\n")
    
    start_idx = 0
    for line in lines:
            if len(line) > 1:
                row.append({"article_id": article_id,
                            "text": "[SKIP]" if len(line) < 1 else line.strip(),
                            "article_path": path,
                            "start_idx": start_idx,
                            "end_idx": end_idx})
            start_idx = end_idx + 1

df = pd.DataFrame(row)
df.to_csv("data/task1_dev.csv")

# %%
