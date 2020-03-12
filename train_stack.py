# %%
import pandas as pd
random_seed = 1956 

all_data = pd.read_csv("data/task2_data.csv")
train = all_data[all_data["source"] == "train"]

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

tfidf = TfidfVectorizer(strip_accents="ascii",
                        analyzer="word",
                        stop_words="english",
                        ngram_range=(1,3),
                        min_df=2)
tfidf = tfidf.fit(all_data["context"])
X_train = tfidf.transform(train["text"])

le = LabelEncoder()
le = le.fit(train["label"])
y_train = le.transform(train["label"])

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=random_seed)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)


# %%
from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingCVClassifier

import numpy as np
import warnings

warnings.simplefilter('ignore')

clf1 = LinearSVC(C=10, max_iter=6000)
clf2 = AdaBoostClassifier(learning_rate=.1, n_estimators=100)
clf3 = RandomForestClassifier(n_jobs=-1, max_depth=6, n_estimators=50)
lr = LogisticRegression(max_iter=6000)

sclf = StackingCVClassifier(classifiers = [clf1, clf2, clf3],
                            meta_classifier=lr,
                            random_state=random_seed)
params = {}

grid = GridSearchCV(estimator=sclf, 
                    param_grid=params, 
                    cv=10,
                    refit=True,
                    verbose=True,
                    n_jobs=-1)

print("training")

grid.fit(X_train_resampled, y_train_resampled)

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)

print("saving....")
import pickle
pickle.dump(grid, open("stack_ensemble6.pkl","wb"))

pickle.dump(tfidf, open("tidf.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print('saved')