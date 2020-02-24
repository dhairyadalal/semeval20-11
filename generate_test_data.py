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
    
    for line in lines:
        if len(line) > 1:
            row.append({"article_id": article_id,
                        "text": line.strip(),
                        "article_path": path})
    
df = pd.DataFrame(row)
df.to_csv("data/task1_test.csv")

# %%
