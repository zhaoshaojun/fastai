# **Important: This notebook will only work with fastai-0.7.x. Do not try to run any fastai-1.x code from this path in the repository because it will load fastai-0.7.x**

# ## video 8, 0:0:0

# ## Using SGD on MNIST

# ## Imports and data

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.imports import *
# from fastai.torch_imports import *
# from fastai.io import *

path = 'data/mnist/'

# Let's download, unzip, and format the data.

import os
os.makedirs(path, exist_ok=True)

# +
URL='http://deeplearning.net/data/mnist/'
FILENAME='mnist.pkl.gz'

def load_mnist(filename):
    return pickle.load(gzip.open(filename, 'rb'), encoding='latin-1')


# -

from fastai.io import *

get_data(URL+FILENAME, path+FILENAME)
((x, y), (x_valid, y_valid), _) = load_mnist(path+FILENAME)

type(x), x.shape, type(y), y.shape

# ### Normalize

# Many machine learning algorithms behave better when the data is *normalized*, that is when the mean is 0 and the standard deviation is 1. We will subtract off the mean and standard deviation from our training set in order to normalize the data:

# +
mean = x.mean()
std = x.std()

x=(x-mean)/std
mean, std, x.mean(), x.std()
# -

# Note that for consistency (with the parameters we learn when training), we subtract the mean and standard deviation of our training set from our validation set. 

x_valid = (x_valid-mean)/std
x_valid.mean(), x_valid.std()


# ### Look at the data

# In any sort of data science work, it's important to look at your data, to make sure you understand the format, how it's stored, what type of values it holds, etc. To make it easier to work with, let's reshape it into 2d images from the flattened 1d format.

# #### Helper methods

def show(img, title=None):
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)


def plots(ims, figsize=(12,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')


# #### Plots 

x_valid.shape

x_imgs = np.reshape(x_valid, (-1,28,28)); x_imgs.shape

show(x_imgs[0], y_valid[0])

y_valid.shape

# It's the digit 3!  And that's stored in the y value:

y_valid[0]

# We can look at part of an image:

x_imgs[0,10:15,10:15]

show(x_imgs[0,10:15,10:15])

plots(x_imgs[:8], titles=y_valid[:8])

# ## Neural Networks

# ## Neural Net for Logistic Regression in PyTorch

import cv2

# +
from fastai.metrics import *
from fastai.model import *
from fastai.dataset import *

import torch.nn as nn
# -

# We will begin with the highest level abstraction: using a neural net defined by PyTorch's Sequential class.  

net = nn.Sequential(
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.LogSoftmax()
)# .cuda()

net = nn.Sequential(
    nn.Linear(28*28, 10),
    nn.LogSoftmax()
)# .cuda()

# Each input is a vector of size `28*28` pixels and our output is of size `10` (since there are 10 digits: 0, 1, ..., 9). 
#
# We use the output of the final layer to generate our predictions.  Often for classification problems (like MNIST digit classification), the final layer has the same number of outputs as there are classes.  In that case, this is 10: one for each digit from 0 to 9.  These can be converted to comparative probabilities.  For instance, it may be determined that a particular hand-written image is 80% likely to be a 4, 18% likely to be a 9, and 2% likely to be a 3.

md = ImageClassifierData.from_arrays(path, (x,y), (x_valid, y_valid))

loss=nn.NLLLoss()
metrics=[accuracy]
opt=optim.Adam(net.parameters())
# opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9, weight_decay=1e-3)

# ### Fitting the model

# *Fitting* is the process by which the neural net learns the best parameters for the dataset.

fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)

t = [o.numel() for o in net.parameters()]
t, sum(t)

preds = predict(net, md.val_dl)

preds.shape

# **Question**: Why does our output have length 10 (for each image)?

import pandas as pd
pd.DataFrame(preds[:5,]).T

plt.pcolor(preds[:5,], edgecolor='k', linewidths=4)

preds.argmax(axis=1)[:5]

preds_cat = preds.argmax(1)

# Let's check how accurate this approach is on our validation set. You may want to compare this against other implementations of logistic regression, such as the one in sklearn. In our testing, this simple pytorch version is faster and more accurate for this problem!

np.mean(preds_cat == y_valid)

# Let's see how some of our predictions look!

plots(x_imgs[:8], titles=preds_cat[:8])


# ## video 8, 1:18:52, again Video 8, 0:17:00

# ## Defining Logistic Regression Ourselves

# Above, we used pytorch's `nn.Linear` to create a linear layer.  This is defined by a matrix multiplication and then an addition (these are also called `affine transformations`).  Let's try defining this ourselves.
#
# Just as Numpy has `np.matmul` for matrix multiplication (in Python 3, this is equivalent to the `@` operator), PyTorch has `torch.matmul`.  
#
# Our PyTorch class needs two things: constructor (says what the parameters are) and a forward method (how to calculate a prediction using those parameters)  The method `forward` describes how the neural net converts inputs to outputs.
#
# In PyTorch, the optimizer knows to try to optimize any attribute of type **Parameter**.

def get_weights(*dims): return nn.Parameter(torch.randn(dims)/dims[0])


def softmax(x): return torch.exp(x)/(torch.exp(x).sum(dim=1)[:,None])


# +
# def softmax(x): return torch.exp(x)/(torch.exp(x).sum(dim=0))
# -

class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_w = get_weights(28*28, 10)  # Layer 1 weights
        self.l1_b = get_weights(10)         # Layer 1 bias

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = (x @ self.l1_w) + self.l1_b  # Linear Layer
        x = torch.log(softmax(x)) # Non-linear (LogSoftmax) Layer
        return x


# We create our neural net and the optimizer.  (We will use the same loss and metrics from above).

net2 = LogReg()# .cuda()
opt=optim.Adam(net2.parameters())

fit(net2, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)

# ## Video 9, 0:29:0

dl = iter(md.trn_dl)

xmb,ymb = next(dl)

xmb.shape, xmb

vxmb = Variable(xmb) # .cuda())
vxmb.shape, vxmb

preds = net2(vxmb).exp(); preds[:3]

preds_cat = preds.data.max(1)[1]; preds_cat

# Let's look at our predictions on the first eight images:

preds_cat = predict(net2, md.val_dl).argmax(1)
plots(x_imgs[:8], titles=preds_cat[:8])

np.mean(preds_cat == y_valid)


# ## Video 9, 00:47:00

# ## Broadcasting

# ## Video 9, 1:18:00, again Video 10, 0:08:40

# ## Writing Our Own Training Loop

# As a reminder, this is what we did above to write our own logistic regression class (as a pytorch neural net):

# +
# Our code from above
class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_w = get_weights(28*28, 10)  # Layer 1 weights
        self.l1_b = get_weights(10)         # Layer 1 bias

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x @ self.l1_w + self.l1_b 
        return torch.log(softmax(x))

net2 = LogReg()# .cuda()
opt=optim.Adam(net2.parameters())

fit(net2, md, n_epochs=1, crit=loss, opt=opt, metrics=metrics)
# -

# Above, we are using the fastai method `fit` to train our model.  Now we will try writing the training loop ourselves.
#
# **Review question:** What does it mean to train a model?

# We will use the LogReg class we created, as well as the same loss function, learning rate, and optimizer as before:

net2 = LogReg()# .cuda()
loss=nn.NLLLoss()
learning_rate = 1e-3
optimizer=optim.Adam(net2.parameters(), lr=learning_rate)

# md is the ImageClassifierData object we created above.  We want an iterable version of our training data (**question**: what does it mean for something to be iterable?):

dl = iter(md.trn_dl) # Data loader

# First, we will do a **forward pass**, which means computing the predicted y by passing x to the model.

xt, yt = next(dl)
y_pred = net2(Variable(xt)) # .cuda())

# We can check the loss:

l = loss(y_pred, Variable(yt)) # .cuda())
print(l)

l = loss(y_pred, yt) # .cuda())
print(l)

# We may also be interested in the accuracy.  We don't expect our first predictions to be very good, because the weights of our network were initialized to random values.  Our goal is to see the loss decrease (and the accuracy increase) as we train the network:

np.mean(to_np(y_pred).argmax(axis=1) == to_np(yt))

# Now we will use the optimizer to calculate which direction to step in.  That is, how should we update our weights to try to decrease the loss?
#
# Pytorch has an automatic differentiation package ([autograd](http://pytorch.org/docs/master/autograd.html)) that takes derivatives for us, so we don't have to calculate the derivative ourselves!  We just call `.backward()` on our loss to calculate the direction of steepest descent (the direction to lower the loss the most).

# +
# Before the backward pass, use the optimizer object to zero all of the
# gradients for the variables it will update (which are the learnable weights
# of the model)
optimizer.zero_grad()

# Backward pass: compute gradient of the loss with respect to model parameters
l.backward()

# Calling the step function on an Optimizer makes an update to its parameters
optimizer.step()
# -

# Now, let's make another set of predictions and check if our loss is lower:

xt, yt = next(dl)
y_pred = net2(Variable(xt)) # .cuda())

l = loss(y_pred, Variable(yt)) # .cuda())
print(l)

l.item()

# Note that we are using **stochastic** gradient descent, so the loss is not guaranteed to be strictly better each time.  The stochasticity comes from the fact that we are using **mini-batches**; we are just using 64 images to calculate our prediction and update the weights, not the whole dataset.

np.mean(to_np(y_pred).argmax(axis=1) == to_np(yt))

# If we run several iterations in a loop, we should see the loss decrease and the accuracy increase with time.

for t in range(100):
    xt, yt = next(dl)
    y_pred = net2(Variable(xt)) # .cuda())
    l = loss(y_pred, Variable(yt)) # .cuda())
    
    if t % 10 == 0:
        accuracy = np.mean(to_np(y_pred).argmax(axis=1) == to_np(yt))
        # print("loss: ", l.data[0], "\t accuracy: ", accuracy)
        print("loss: ", l.item(), "\t accuracy: ", accuracy)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()


# ### Put it all together in a training loop

def score(x, y):
    y_pred = to_np(net2(V(x)))
    return np.sum(y_pred.argmax(axis=1) == to_np(y))/len(y_pred)


# +
net2 = LogReg() # .cuda()
loss=nn.NLLLoss()
learning_rate = 1e-2
optimizer=optim.SGD(net2.parameters(), lr=learning_rate)

losses=[]
for epoch in range(10):
    # dl = iter(md.trn_dl)
    # for t in range(len(dl)):
    for t in md.trn_dl:
        # Forward pass: compute predicted y and loss by passing x to the model.
        xt, yt = t
        y_pred = net2(V(xt))
        l = loss(y_pred, V(yt))
        losses.append(l)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        l.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
    
    # val_dl = iter(md.val_dl)
    val_scores = [score(x, y) for x, y in md.val_dl]
    print(epoch, to_np(losses[epoch]).flat[0], np.mean(val_scores))
# -

# ## Video 10, 00:18:24

# ## Stochastic Gradient Descent

# Nearly all of deep learning is powered by one very important algorithm: **stochastic gradient descent (SGD)**. SGD can be seeing as an approximation of **gradient descent (GD)**. In GD you have to run through all the samples in your training set to do a single itaration. In SGD you use only a subset of training samples to do the update for a parameter in a particular iteration. The subset used in each iteration is called a batch or minibatch.
#
# Now, instead of using the optimizer, we will do the optimization ourselves!

# +
net2 = LogReg() # .cuda()
loss_fn=nn.NLLLoss()
lr = 1e-2
w,b = net2.l1_w,net2.l1_b

losses=[]
for epoch in range(10):
    # dl = iter(md.trn_dl)
    for t in md.trn_dl:
        xt, yt = t
        y_pred = net2(V(xt))
        l = loss(y_pred, Variable(yt)) # .cuda())
        losses.append(l)

        # Backward pass: compute gradient of the loss with respect to model parameters
        l.backward()
        w.data -= w.grad.data * lr
        b.data -= b.grad.data * lr
        
        w.grad.data.zero_()
        b.grad.data.zero_()   

    # val_dl = iter(md.val_dl)
    val_scores = [score(x, y) for x, y in md.val_dl]
    print(epoch, to_np(losses[epoch]).flat[0], np.mean(val_scores))
    # print(epoch, np.mean(val_scores))
# -
# ## Video 10, 00:29:00 - 00:33:00 Going backwards

# ## Video 10, 00:33:00 Deep networks

# ## Video 10, 00:42:00 - 00:58:00, Regularization

# ## Video 10, 00:58:00 - 1:02:00, Big vs. Small Models

# ## Video 10, 1:02:00 IMDB


