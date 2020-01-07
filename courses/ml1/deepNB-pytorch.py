# +
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.nlp import *
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.utils import shuffle
# -

# ## create dataset

# PATH = 'data/aclImdb/'
PATH = '/Users/shaojun/c/fastai/courses/ml1/data/aclImdb/'
names = ['neg','pos']
names1 = ['neg', 'pos_']
names2 = ['neg_', 'pos']

# ! ls {PATH}train

trn1,trn1_y = texts_labels_from_folders(f'{PATH}train',names1)
val1,val1_y = texts_labels_from_folders(f'{PATH}test',names1)

trn2,trn2_y = texts_labels_from_folders(f'{PATH}train',names2)
val2,val2_y = texts_labels_from_folders(f'{PATH}test',names2)

len(trn1), len(trn1_y), len(trn2), len(trn2_y)

len(val1), len(val2)

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
    np.random.seed(123)
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


trn, trn_y = sample(trn1, trn1_y, trn2, trn2_y, 64*100)
val, val_y = sample(val1, val1_y, val2, val2_y, 64*20)

# +
# trn,trn_y = texts_labels_from_folders(f'{PATH}train',names)
# val,val_y = texts_labels_from_folders(f'{PATH}test',names)
# -

# ## create vectors and vocab

veczr = CountVectorizer(tokenizer=tokenize)

trn_term_doc = veczr.fit_transform(trn)

val_term_doc = veczr.transform(val)

vocab = veczr.get_feature_names(); vocab[5000:5005]

vocab[:5]

len(vocab)

trn_term_doc = trn_term_doc.sign()

val_term_doc = val_term_doc.sign()

trn_term_doc.shape, val_term_doc.shape


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

b

val_term_doc

(val_term_doc @ r.T).shape

val_term_doc @ r.T

val_y

pre_preds = val_term_doc @ r.T + b
preds = pre_preds.T>0
(preds==val_y).mean()

pre_preds = val_term_doc @ np.stack([np.log(p), np.log(q)]).T + b

pre_preds

preds = pre_preds.T[0] > pre_preds.T[1]
nb_score = (preds==val_y).mean()
nb_score

type(val_term_doc)

val_term_doc[0]

xx = val_term_doc[0]

xx.toarray().shape

# ## Logistic regression (sklearn)

# Here is how we can fit logistic regression where the features are the unigrams.

LogisticRegression

m = LogisticRegression(C=1e8, dual=False, max_iter=1000)
m.fit(x, y)
preds = m.predict(val_term_doc)
lr_score = (preds==val_y).mean()
lr_score

# ...and the regularized version

m = LogisticRegression(C=1.0, dual=False, max_iter=1000)
m.fit(x, y)
preds = m.predict(val_term_doc)
lr_score2 = (preds==val_y).mean()
lr_score2


# ## Logistic regression (PyTorch)

def binary_loss(pred, y):
    pred = pred.clamp(-10, 10)
    # y2 = torch.max(y,axis=1)[0]
    y2 = y
    p = torch.exp(pred) / (1+torch.exp(pred))
    result = torch.mean(-(y2 * torch.log(p) + (1-y2)*torch.log(1-p)))
    # return result.reshape(1, -1)
    return result


def score(pred, y):
    return np.sum(to_np((pred > 0) == y))


class SimpleNB2(nn.Module):
    def __init__(self, nf, ny):
        super().__init__()
        self.w = nn.Embedding(nf, ny)
        # self.w = nn.Embedding(nf+1, ny)
        # self.w.weight.data.uniform_(-1, 1)
        self.w.weight.data = torch.FloatTensor(r)[0].reshape(-1,1)
        # self.r = nn.Embedding(nf, ny)

    def forward(self, feat_idx):
        idx = feat_idx.nonzero()[1]
        v = self.w(V(idx))
        x = v.sum()
        return x


# +
net_a = SimpleNB2(len(vocab),1)

loss = binary_loss
# loss = torch.nn.CrossEntropyLoss
lr = 1e-4
losses=[]
# -



idx = trn_term_doc[0].nonzero()[1]
idx

net_a.w(V(idx)).sum()

net_a(trn_term_doc[-1])

trn_y[-1]

pred = net_a(trn_term_doc[-1])

y = trn_y[-1]

binary_loss(pred, y)

score(net_a(trn_term_doc[-1]), trn_y[-1])

trn_term_doc[0] @ r.T

trn_term_doc[-1] @ r.T



net_a(trn_term_doc[0])

(net_a(trn_term_doc[0]) > 0) == trn_y[0]

lr

import os
filename = 'acc.txt'
try:
    os.remove(filename)
    print('removed')
except:
    print('pass')
    pass

ii = 1

ii = 67

_x = trn_term_doc[ii]
_y = trn_y[ii]

_y_pred = net_a(_x)
_y_pred

l = loss(_y_pred, V(_y))
# l = loss(yt, y_pred)
# loss_list.append(l)
# print(f'{index}, {l}, {datetime.now().time()}')



l

l.backward()

l

net_a.w.weight.grad.data

# +
net_a.w.weight.data -= net_a.w.weight.grad.data * lr
# net2.b.data -= net2.b.grad.data * lr

net_a.w.weight.grad.data.zero_()
# -

net_a.w.weight.data

# ## Train

train_loss_list = []
val_loss_list = []
val_acc_list = []
train_acc_list = []

# +
net_a = SimpleNB2(len(vocab),1)

loss = binary_loss
# loss = torch.nn.CrossEntropyLoss
lr = 1e-3
wd = 1e-8
# -

trn_scores = []
for x, y in tqdm(zip(trn_term_doc, trn_y), total=trn_term_doc.shape[0]):
    trn_scores.append(score(net_a(x),y))
print(np.mean(to_np(trn_scores)))

val_scores = []
for x, y in tqdm(zip(val_term_doc, val_y), total=val_term_doc.shape[0]):
    val_scores.append(score(net_a(x),y))
print(np.mean(to_np(val_scores)))

if False:
    train_acc_scores = []
    for x, y in tqdm(zip(trn_term_doc, trn_y), total=trn_term_doc.shape[0]):
        train_acc_scores.append(score(net_a(x),y))
    l3 = np.mean(to_np(train_acc_scores))
    print(l3)

    acc_scores = []
    for x, y in tqdm(zip(val_term_doc, val_y), total=val_term_doc.shape[0]):
        acc_scores.append(score(net_a(x),y))
    l4 = np.mean(to_np(acc_scores))
    print(l4)

# +
print(f'lr={lr},wd={wd}')
f = open(filename, 'a')

from datetime import datetime
train_loss_list = []
val_loss_list = []
val_acc_list = []
train_acc_list = []

# loss_list = [0]
loss_list = []
for epoch in range(1000):
    # learning rate annealing
    if epoch == 10:
        lr /= 10
    if epoch == 20:
        lr /= 10
    # eval
    if epoch % 1 == 0:
        train_scores = []
        for x, y in tqdm(zip(trn_term_doc, trn_y), total=trn_term_doc.shape[0]):
            w2 = 0
            for p in net_a.parameters():
                w2 += (p**2).sum()
            l = loss(net_a(x), V(y)) + wd * w2
            train_scores.append(l)
        l1 = np.mean(to_np(train_scores))
        train_loss_list.append(l1)

        val_scores = []
        for x, y in tqdm(zip(val_term_doc, val_y), total=val_term_doc.shape[0]):
            w2 = 0
            for p in net_a.parameters():
                w2 += (p**2).sum()
            l = loss(net_a(x), V(y)) + wd * w2
            val_scores.append(l)
        l2 = np.mean(to_np(val_scores))
        val_loss_list.append(l2)

        train_acc_scores = []
        for x, y in tqdm(zip(trn_term_doc, trn_y), total=trn_term_doc.shape[0]):
            train_acc_scores.append(score(net_a(x),y))
        l3 = np.mean(to_np(train_acc_scores))
        train_acc_list.append(l3)

        acc_scores = []
        for x, y in tqdm(zip(val_term_doc, val_y), total=val_term_doc.shape[0]):
            acc_scores.append(score(net_a(x),y))
        l4 = np.mean(to_np(acc_scores))
        val_acc_list.append(l4)

        # print(f'epoch={epoch}, score={np.mean(val_scores)}')
        # print(f'epoch={epoch}, score={l2}')
        print(f'epoch={epoch}, train-loss={l1}')
        print(f'epoch={epoch}, valid-loss={l2}')
        print(f'epoch={epoch}, train-acc={l3}')
        print(f'epoch={epoch}, valid-acc={l4}')
        f.write(f"{epoch}\t{lr}\t{wd}\t{l1}\t{l2}\t{l3}\t{l4}\t{nb_score}\t{lr_score}\t{lr_score2}\n")
        f.flush()

    print('')
    print('epoch:', epoch)
    print('time:', datetime.now())
    shuffle_x, shuffle_y = shuffle(trn_term_doc, trn_y)

    batch_size = 16
    batch_loss = []
    for _x, _y in tqdm(zip(shuffle_x, shuffle_y), total=shuffle_x.shape[0]):
        if len(batch_loss) == batch_size:
            w2 = 0
            for p in net_a.parameters():
                w2 += (p**2).sum()
            l = 0
            for one_loss in batch_loss:
                l += one_loss
            l = 1 / batch_size + wd * w2
            batch_loss = []
            # l = loss(yt, y_pred)
            loss_list.append(l)
            # print(f'{index}, {l}, {datetime.now().time()}')

            # Backward pass:
            # compute gradient of the loss with respect to
            # model parameters
            l.backward()
            net_a.w.weight.data -= net_a.w.weight.grad.data * lr
            # net2.b.data -= net2.b.grad.data * lr

            net_a.w.weight.grad.data.zero_()
            # net2.b.grad.data.zero_()

        one_loss = loss(net_a(_x), V(_y))
        batch_loss.append(one_loss)

f.close()
# -
len(loss_list)


import matplotlib.pyplot as plt
import pandas as pd

len(train_loss_list)

length=len(train_loss_list)
df = pd.DataFrame({
    'train':train_loss_list[:length],
    'valid':val_loss_list[:length],
    'train_acc':train_acc_list[:length],
    'valid_acc':val_acc_list[:length]
})

df

df[['train','valid']].plot(subplots=True)

df.plot(subplots=True)

plt.plot(loss_list[:1280])

plt.plot(loss_list)

plt.plot(loss_list)

# ## Deep NB


