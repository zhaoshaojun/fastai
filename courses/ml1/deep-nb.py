# +
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.nlp import *
from sklearn.linear_model import LogisticRegression
# -

# ## create dataset

PATH='data/aclImdb/'
names = ['neg','pos']
names1 = ['neg', 'pos_']
names2 = ['neg_', 'pos']

# ! ls {PATH}train

trn1,trn1_y = texts_labels_from_folders(f'{PATH}train',names1)
val1,val1_y = texts_labels_from_folders(f'{PATH}test',names1)

trn2,trn2_y = texts_labels_from_folders(f'{PATH}train',names2)
val2,val2_y = texts_labels_from_folders(f'{PATH}test',names2)

len(trn1), len(trn1_y), len(trn2), len(trn2_y)

assert (trn1_y==0).all()
(trn1_y==0).all()

assert (trn1_y==0).all()
(val1_y==0).all()

assert (trn2_y==1).all()
(trn2_y==1).all()

assert (val2_y==1).all()
(val2_y==1).all()


def sample_util(data, label, n):
    assert len(data) == label.shape[0]
    idx = np.random.choice(range(len(data)),n)
    data_new = [data[i] for i in idx]
    label_new = label[idx]
    return data_new, label_new


def sample(data1, label1, data2, label2, n):
    t1, t2 = sample_util(data1, label1, n)
    t3, t4 = sample_util(data2, label2, n)
    data = t1 + t3
    label = np.concatenate((t2, t4))
    return data, label


trn, trn_y = sample(trn1, trn1_y, trn2, trn2_y, 64*10)
val, val_y = sample(val1, val1_y, val2, val2_y, 64*20)

# +
# trn,trn_y = texts_labels_from_folders(f'{PATH}train',names)
# val,val_y = texts_labels_from_folders(f'{PATH}test',names)
# -

# ## create vectors and vocab

veczr = CountVectorizer(tokenizer=tokenize)

trn_term_doc = veczr.fit_transform(trn)

val_term_doc = veczr.transform(val)

vocab[:5]

vocab = veczr.get_feature_names(); vocab[5000:5005]

len(vocab)


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

val_term_doc.shape[1]

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

m = LogisticRegression(C=0.1, dual=False, max_iter=1000)
m.fit(x, y)
preds = m.predict(val_term_doc)
(preds==val_y).mean()

# ## Logistic regression (PyTorch)

# +
from fastai.metrics import *
from fastai.model import *
from fastai.dataset import *
from fastai.nlp import *

import torch.nn as nn


# -

class MySimpleNB(nn.Module):
    def __init__(self, nf, ny, w_adj=0.4, r_adj=10):
        super().__init__()
        self.w_adj,self.r_adj = w_adj,r_adj
        self.w = nn.Embedding(nf+1, 2, padding_idx=0)
        self.w.weight.data.uniform_(-0.1,0.1)
        # self.r = nn.Embedding(nf+1, ny)
        
    def forward(self, feat_idx):
        w = self.w(feat_idx)
        # r = self.r(feat_idx)
        # x = ((w+self.w_adj)*r/self.r_adj).sum(1)
        # x = (w+self.w_adj)*r/self.r_adj
        return F.softmax(w)


len(vocab)

sl=val_term_doc.shape[1]
sl

sl=val_term_doc.shape[1]
md = TextClassifierData.from_bow(
    trn_term_doc, trn_y,
    val_term_doc, val_y,
    sl
)

trn_term_doc.shape

np.stack([trn_y, 1 - trn_y]).T.shape

dl = iter(md.trn_dl)

net2 = MySimpleNB(len(vocab), 2)
# loss = nn.NLLLoss()
loss = torch.nn.CrossEntropyLoss()
lr = 1e-2
losses=[]

t = next(dl)
xt, _a, _b, yt = t
y_pred = net2(_b)
y_pred.shape, y_pred

y_pred.sum()

t = next(dl)
xt, _a, _b, yt = t
y_pred = net2(_b)
y_pred.shape, y_pred

l = loss(y_pred, np.argmax(yt, axis=1))

l

l.backward()

# +
# net2.w.weight.data -= net2.w.weight.grad.data * lr
# -

lr

net2.w.weight.data

for i in net2.w.weight.grad.data:
    for j in i:
        if j != 0:
            print(j)





from datetime import datetime

len(md.trn_dl)


def score(x, y):
    y_pred = to_np(net2(V(x)))
    return np.sum(y_pred.argmax(axis=1) == to_np(y).argmax(axis=1))/len(y_pred)


from tqdm import notebook

# +
net2 = MySimpleNB(len(vocab), 2)
# loss = nn.NLLLoss()
loss = torch.nn.CrossEntropyLoss()
lr = 1e-2
train_list = []
val_list = []
loss_list = []

print(f'lr={lr}')
for epoch in range(20):
    print('')
    print('epoch:', epoch)
    print('time:', datetime.now().time())
    for index, t in tqdm(enumerate(md.trn_dl), total=len(md.trn_dl)):
        xt, _a, _b, yt = t
        y_pred = net2(V(_b))
        l = loss(y_pred, V(np.argmax(yt, axis=1)))
        # l = loss(yt, y_pred)
        loss_list.append(l)
        # print(f'{index}, {l}, {datetime.now().time()}')

        # Backward pass: 
        # compute gradient of the loss with respect to 
        # model parameters
        l.backward()
        net2.w.weight.data -= net2.w.weight.grad.data * lr
        # net2.b.data -= net2.b.grad.data * lr
        
        net2.w.weight.grad.data.zero_()
        # net2.b.grad.data.zero_()   

    if epoch % 2 == 0:
        train_scores = [loss(net2(_b), np.argmax(y, axis=1)) 
                      for x, _a, _b, y in md.trn_dl]
        l2 = np.mean(to_np(train_scores))
        train_list.append(l2)

        val_scores = [score(_b, y) for x, _a, _b, y in md.val_dl]
        l3 = np.mean(to_np(val_scores))
        val_list.append(l3)

        # print(f'epoch={epoch}, score={np.mean(val_scores)}')
        print(f'epoch={epoch}, score={l2}')
        print(f'epoch={epoch}, score={l3}')
# -

import matplotlib.pyplot as plt

plt.plot(train_list)

plt.plot(val_list)

plt.plot(loss_list)

net2.r

net2.w

val_dl = iter(md.val_dl)
val_scores = [score(_b, y) for x, _a, _b, y in val_dl]

np.mean(val_scores)

val_scores

val_dl = iter(md.val_dl)

x, _a, _b, y = next(val_dl)



score(x, y)

w = net2.w(x)

r = net2.r(x)

x = ((w+net2.w_adj)*r/net2.r_adj).sum(1)

x

result = F.softmax(x)
result

result.shape

to_np(result).argmax(axis=1) == to_np(y).argmax(axis=1)

to_np(y)

# ## Deep NB


