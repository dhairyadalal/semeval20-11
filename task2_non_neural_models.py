#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report


dat = pd.read_csv("datasets/task_2_data.csv")
le = LabelEncoder()

train_dat = dat[dat["source"]=="train"]
dev = dat[dat["source"] != "train" ]

train_dat["encoded_label"] = le.fit_transform(train_dat["label"])
random_seed = 1988

train, val = train_test_split(train_dat, 
                              stratify=train_dat["label"], 
                              random_state=random_seed, 
                              test_size=.1)

le = le.fit(train["label"])
train["encoded_label"] = le.fit_transform(train["label"])

#%%
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train("datasets/all_text_corpus.txt", 
                vocab_size=15000, 
                min_frequency=2,
                show_progress=True)
MAX_LEN = 224

def pad_tokenize(input: pd.Series, pad_size: int, pad_value: int) -> list:
    ids = []
    for text in input:
        padded_seq = [pad_value] * pad_size
        input_ids = tokenizer.encode(text).ids
        padded_seq[: len(input_ids)] = input_ids
        ids.append(padded_seq)
    return np.array(ids)
    
train_inputs = pad_tokenize(train["text"], MAX_LEN, 0)
train_labels = train["encoded_label"].tolist()

val_inputs = pad_tokenize(val["text"],MAX_LEN,0)
val_labels = np.array(val["label"].tolist())

all_train_inputs = pad_tokenize(train_dat["text"], MAX_LEN, 0)
all_train_labels = np.array(train_dat["label"].tolist())

dev_inputs = pad_tokenize(dev["text"], MAX_LEN, 0)

ros = RandomOverSampler(random_state=random_seed)
X_train_resampled, y_train_resampled = ros.fit_resample(train_inputs, 
                                                        train_labels)

X_train_all_resampled, y_train_all_resampled = ros.fit_resample(all_train_inputs,
                                                                all_train_labels)

# %%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# clf = RandomForestClassifier(random_state=random_seed, 
#                              verbose=True, n_jobs=-1)

# param_grid = {'n_estimators': [100, 200, 300, 400, 500],
#               'max_features': ['auto', 'sqrt', 'log2'],
#               'max_depth' : 'sqrt', 'log2'[4,5,6,7,8],
#               'criterion' :['gini', 'entropy']}

# CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)
# CV_clf.fit(train_inputs, train_labels)


# Best Params:

# CV_clf.best_params_
# {'criterion': 'gini',
#  'max_depth': 8,
#  'max_features': 'auto',
#  'n_estimators': 200}

# %%
#rfclf = CV_clf.best_estimator_

# %%

val_preds = rfclf.predict(val_inputs)
val_preds = le.inverse_transform(val_preds)

print(classification_report(val_preds, val_labels))

# %%

# %%
dev_preds = rfclf.predict(dev_inputs)

dev_preds = le.inverse_transform(dev_preds)

# %%
#%%
with open("datasets/dev-task-TC-template.out", "r") as f:
    lines = f.readlines()
    
print(lines[0]) 
# %%
final = []
for i, line in enumerate(lines):
    pred = dev_preds[i].strip()
    line = line.replace("?", pred)
    final.append(line)


# %%
with open("submissions/random_forest_base.txt", "w") as f:
    for line in final:
        f.write(line)

#%%

clf = RandomForestClassifier(random_state=random_seed, 
                             verbose=True, n_jobs=-1)

param_grid = {'n_estimators': [100, 200, 300, 400, 500],
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth' : [4,5,6,7,8],
              'criterion' :['gini', 'entropy']}

CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)
#CV_clf.fit(X_train_all_resampled, y_train_all_resampled)

# Best Params
# {'criterion': 'entropy',
#  'max_depth': 8,
#  'max_features': 'auto',
#  'n_estimators': 500}

#%%
clf = CV_clf.best_estimator_
dev_preds = clf.predict(dev_inputs)

# %%
from typing import List
def write_t2_preds(preds: List[str], file_name: str):
    template_file = "datasets/dev-task-TC-template.out"

    with open(template_file, "r") as f:
        template_lines = f.readlines()
        
    final = []
    for i, line in enumerate(template_lines):
        pred = preds[i].strip()
        line = line.replace("?", pred)
        final.append(line)
    
    save_path = "submissions/"
    with open(save_path+file_name,"w") as f:
        for line in final:
            f.write(line) 

write_t2_preds(dev_preds, "random_forest_all_train.txt")    

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

all_text = [tokenizer.decode(text).strip("!") for text in X_train_all_resampled]
dev_text = [tokenizer.decode(text).strip("!") for text in dev_inputs]

fit_text = all_text + dev_text

tfidf = TfidfVectorizer(lowercase=True, 
                        stop_words='english', 
                        min_df=2)
tfidf.fit(fit_text)
X = tfidf.fit_transform(all_text)

clf2 = RandomForestClassifier(random_state=random_seed, 
                             verbose=True, 
                             n_jobs=-1)

param_grid = {'n_estimators': [500],
              'max_features': ['auto'],
              'max_depth' : [8],
              'criterion' :['entropy']}

CV2_clf = GridSearchCV(estimator=clf2, param_grid=param_grid, cv=10)
CV2_clf.fit(X, y_train_all_resampled)
#%%
clf2 = CV2_clf.best_estimator_

# %%
X_dev = tfidf.transform(dev_text)

dev_preds = clf2.predict(X_dev)


# %%
write_t2_preds(dev_preds, "random_forest_tfidf.txt")    


# %%
from sklearn.svm import SVC

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
y = le.transform(y_train_all_resampled)

CV3_clf = GridSearchCV(estimator=SVC(verbose=True), 
                       param_grid=tuned_parameters, cv=5)
CV3_clf.fit(X, y)

# %%
# best param: {'C': 1000, 'kernel': 'linear'}
clf3 = CV3_clf.best_estimator_

dev_preds = clf3.predict(X_dev)
dev_preds = le.inverse_transform(dev_preds)

# %%
write_t2_preds(dev_preds, "SVM_classifier_tfidf.txt")    


# %%
