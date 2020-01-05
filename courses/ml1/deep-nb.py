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


# +
# trn, trn_y = sample(trn1, trn1_y, trn2, trn2_y, 64*10)
# val, val_y = sample(val1, val1_y, val2, val2_y, 64*20)
# -

trn,trn_y = texts_labels_from_folders(f'{PATH}train',names)
val,val_y = texts_labels_from_folders(f'{PATH}test',names)

# ## create vectors and vocab

veczr = CountVectorizer(tokenizer=tokenize)

trn_term_doc = veczr.fit_transform(trn)

val_term_doc = veczr.transform(val)

vocab = veczr.get_feature_names(); vocab[5000:5005]

vocab[:5]

len(vocab)

trn_term_doc = trn_term_doc.sign()

val_term_doc = val_term_doc.sign()


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

type(val_term_doc)

val_term_doc[0]

xx = val_term_doc[0]

[vocab[index] for index, i in enumerate(xx.toarray()[0]) if i > 0]

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
    def __init__(self, nf, ny):
        super().__init__()
        self.w = nn.Embedding(nf, ny)
        # self.w = nn.Embedding(nf+1, ny)
        # self.w.weight.data.uniform_(-1, 1)
        self.w.weight.data = torch.FloatTensor(r.tolist()[0])
        self.w.weight.data = self.w.weight.data.reshape(-1, 1)
        # self.r = nn.Embedding(nf, ny)
        
    def forward(self, feat_idx):
        # self.w.weight.data[0] = 0
        idx = feat_idx - 1
        idx2 = [a for a in idx if a >= 0]
        idx3 = np.array(idx2)
        v = self.w(V(idx3))
        # r = self.r(feat_idx)
        # x = ((w+self.w_adj)*r/self.r_adj).sum(1)
        # x = w*r
        x = v.sum(1)
        # return F.softmax(x)
        # return x.reshape(1, -1)
        return x


def binary_loss(pred, y):
    # y2 = torch.max(y,axis=1)[0]
    y2 = np.argmax(y)
    p = torch.exp(pred) / (1+torch.exp(pred))
    result = torch.mean(-(y2 * torch.log(p) + (1-y2)*torch.log(1-p)))
    # return result.reshape(1, -1)
    return result


r.shape



len(vocab)

net3 = MySimpleNB(len(vocab), 1)

net2.w.weight.data.shape

r.shape

xx = torch.FloatTensor([0] + r.tolist()[0]).reshape(-1, 1)

xx.shape

t

for xx in t[0]:
    print(vocab[xx-1])

net3.w

net3.w.weight.data[0]

net3.w.weight.data.shape



embedding = nn.Embedding(10, 1)

input = torch.LongTensor([[1,2,4,0],[4,3,2,9]])

embedding(input)

embedding(input).sum(1)



sl=val_term_doc.shape[1]
sl

sl=val_term_doc.shape[1]
md = TextClassifierData.from_bow(
    trn_term_doc, trn_y,
    val_term_doc, val_y,
    100,
)

# +
# ??TextClassifierData.from_bow
# -

trn_term_doc.shape

i=0

net4 = MySimpleNB(len(vocab), 1)
# loss = nn.NLLLoss()
# loss = torch.nn.CrossEntropyLoss()
loss = binary_loss
lr = 1e-2
losses=[]

net4.w.weight

ii=2

t = md.trn_ds[ii]
ii = ii + 1
xt, _a, _b, yt = t

t

xt.shape, len(vocab)

xt.shape

xt = xt.reshape(1, 200)

xt.shape

vocab[18178], len(vocab)

_a

_a.sum()

_b

yt

md.trn_ds[ii]

xt.shape

for index, idx in enumerate(to_np(xt[0])):
    if idx:
        print(vocab[idx-1])

vocab[0], len(vocab)

xt

net4.w(V(xt)).sum()

xt.shape

xt.shape

y_pred = net4(V(xt[0]))
print(y_pred)
l = binary_loss(y_pred, yt)

y_pred

print(net4.w.weight.grad)

l.shape

l.backward()

_a.shape, _a

for idx, i in enumerate(net4.w.weight.grad.data):
    for j in i:
        if j != 0:
            print(idx, j, net4.w.weight.data[idx])



net4.w.weight.data -= net4.w.weight.grad.data * lr

lr

net4.w.weight.data

net4.w.weight.grad.data

for i in net4.w.weight.grad.data:
    for j in i:
        if j != 0:
            print(j)

net4.w.weight.grad.data.zero_()



from datetime import datetime

len(md.trn_ds)

a1, a2, a3, a4 = md.trn_ds[0]

a1

a2

a3

a4, np.argmax(a4)

torch.max(V(a4))

loss(net4(V(a1)), V(a4))

net4(V(a1))


def score2(x, y):
    y_pred = to_np(net2(V(x)))
    return np.sum(y_pred.argmax(axis=1) == to_np(y).argmax(axis=1))/len(y_pred)


def score(x, y):
    # print(f'x={x}, y={y}')
    y_pred = to_np(net2(V(x))).sum() >= 0
    # print(f'y_pred={y_pred}')
    y2 = np.argmax(y)
    # print(f'y2={y2}')
    return np.sum(y_pred == y2)



from tqdm import notebook

datetime.now()

net2 = MySimpleNB(len(vocab), 1)
# loss = nn.NLLLoss()
# loss = torch.nn.CrossEntropyLoss()
loss = binary_loss
# lr = 1e-0
lr = 1e-3
train_list = []
val_list = []
loss_list = []

val_scores = []
for t in tqdm(md.val_ds, total=len(md.val_ds)):
    x, _a, _b, y = t
    val_scores.append(score(x,y))
np.mean(to_np(val_scores))

print(f'lr={lr}')
for epoch in range(10):
    print('')
    print('epoch:', epoch)
    print('time:', datetime.now())
    loss_list = [0]
    for index, t in tqdm(enumerate(md.trn_ds), total=len(md.trn_ds)):
        xt, _a, _b, yt = t
        y_pred = net2(V(xt))
        l = loss(y_pred, V(yt))
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

    if epoch % 1 == 0:
        train_scores = []
        for t in tqdm(md.trn_ds, total=len(md.trn_ds)):
            x, _a, _b, y = t
            train_scores.append(loss(net2(V(x)), V(y)))
        l2 = np.mean(to_np(train_scores))
        train_list.append(l2)
        
        val_scores = []
        for t in tqdm(md.val_ds, total=len(md.val_ds)):
            x, _a, _b, y = t
            val_scores.append(score(x,y))
        l3 = np.mean(to_np(val_scores))
        val_list.append(l3)

        # print(f'epoch={epoch}, score={np.mean(val_scores)}')
        print(f'epoch={epoch}, score={l2}')
        print(f'epoch={epoch}, score={l3}')

import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({'train':train_list, 'valid':val_list})

df.plot(subplots=True)

plt.plot(loss_list)

# ## Deep NB


