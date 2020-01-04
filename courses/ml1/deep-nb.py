# +
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.nlp import *
from sklearn.linear_model import LogisticRegression
# -

# ## Tokenizing and term document matrix creation

PATH='data/aclImdb/'
names = ['neg','pos']

trn,trn_y = texts_labels_from_folders(f'{PATH}train',names)
val,val_y = texts_labels_from_folders(f'{PATH}test',names)

veczr = CountVectorizer(tokenizer=tokenize)

trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)

vocab = veczr.get_feature_names(); vocab[5000:5005]


# ## Naive Bayes

# We define the **log-count ratio** $r$ for each word $f$:
#
# $r = \log \frac{\text{ratio of feature $f$ in positive documents}}{\text{ratio of feature $f$ in negative documents}}$
#
# where ratio of feature $f$ in positive documents is the number of times a positive document has a feature divided by the number of positive documents.

def pr(y_i):
    p = x[y==y_i].sum(0)
    return p+1


# +
x=trn_term_doc
y=trn_y

p = pr(1)/pr(1).sum()
q = pr(0)/pr(0).sum()
r = np.log(p/q)
b = np.log((y==1).mean() / (y==0).mean())
# -

val_term_doc.shape

x.shape

r.shape

(val_term_doc @ r.T).shape

val_term_doc @ r.T

val_y

pre_preds = val_term_doc @ r.T + b
preds = pre_preds.T>0
(preds==val_y).mean()

pre_preds = val_term_doc @ np.stack([np.log(p), np.log(q)]).T + b

pre_preds

preds = pre_preds.T[0] > pre_preds.T[1]
(preds==val_y).mean()

# ## Logistic regression (sklearn)

# Here is how we can fit logistic regression where the features are the unigrams.

LogisticRegression

m = LogisticRegression(C=1e8, dual=False, max_iter=1000)
m.fit(x, y)
preds = m.predict(val_term_doc)
(preds==val_y).mean()

# ...and the regularized version

m = LogisticRegression(C=0.01, dual=False, max_iter=1000)
m.fit(x, y)
preds = m.predict(val_term_doc)
(preds==val_y).mean()

# ## Logistic regression (PyTorch)



# ## Deep NB


