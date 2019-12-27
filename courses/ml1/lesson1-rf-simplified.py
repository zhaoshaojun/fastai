# ## Imports

# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline

# +
from fastai.imports import *
from fastai.structured import *

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
# -

PATH = "data/bulldozers/"

# !ls {PATH}

# ## The data

df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, 
                     parse_dates=["saledate"])

df_raw.SalePrice = np.log(df_raw.SalePrice)

# ### Initial processing

try:
    m = RandomForestRegressor(n_estimators=10, n_jobs=-1)
    # The following code is supposed to fail due to string values in the input data
    m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
    assert False
except Exception as e:
    print(f"something went wrong: {e}")

# This dataset contains a mix of **continuous** and **categorical** variables.
#
# The following method extracts particular date fields from a complete datetime for the purpose of constructing categoricals.  You should always consider this feature extraction step when working with date-time. Without expanding your date-time into these additional fields, you can't capture any trend/cyclical behavior as a function of time at any of these granularities.

add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()

# The categorical variables are currently stored as strings, which is inefficient, and doesn't provide the numeric coding required for a random forest. Therefore we call `train_cats` to convert strings to pandas categories.

train_cats(df_raw)

# We can specify the order to use for categorical variables if we wish:

df_raw.UsageBand.cat.categories

df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)

# Normally, pandas will continue displaying the text categories, while treating them as numerical data internally. Optionally, we can replace the text categories with numbers, which will make this variable non-categorical, like so:.

df_raw.UsageBand = df_raw.UsageBand.cat.codes

# We're still not quite done - for instance we have lots of missing values, which we can't pass directly to a random forest.

# But let's save this file for now, since it's already in format can we be stored and accessed efficiently.

os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/bulldozers-raw')

# !ls -lrth tmp/bulldozers-raw

# ## Random Forests

# ### Pre-processing

# In the future we can simply read it from this fast format.

import feather
df_raw = feather.read_dataframe('tmp/bulldozers-raw')

# We'll replace categories with their numeric codes, handle missing continuous values, and split the dependent variable into a separate variable.

df, y, nas = proc_df(df_raw, 'SalePrice')

nas

# ## overfit

# We now have something we can pass to a random forest!

m = RandomForestRegressor(n_estimators=10, n_jobs=-1)
m.fit(df, y)
m.score(df,y)


# Wow, an r^2 of 0.98 - that's great, right? Well, perhaps not...
#
# Possibly **the most important idea** in machine learning is that of having separate training & validation data sets. As motivation, suppose you don't divide up your data, but instead use all of it.  And suppose you have lots of parameters:
#
# <img src="images/overfitting2.png" alt="" style="width: 70%"/>
# <center>
# [Underfitting and Overfitting](https://datascience.stackexchange.com/questions/361/when-is-a-model-underfitted)
# </center>
#
# The error for the pictured data points is lowest for the model on the far right (the blue curve passes through the red points almost perfectly), yet it's not the best choice.  Why is that?  If you were to gather some new data points, they most likely would not be on that curve in the graph on the right, but would be closer to the curve in the middle graph.
#
# This illustrates how using all our data can lead to **overfitting**. A validation set helps diagnose this problem.

# +
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


# -

# ## Base model

# Let's try our model again, this time with separate training and validation sets.

# +
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def get_scores(m, config=None):
    res = {
        'config':[config],
        'rmse_train': [rmse(m.predict(X_train), y_train)], 
        'rmse_dev': [rmse(m.predict(X_valid), y_valid)],
        'r^2_train': [m.score(X_train, y_train)], 
        'r^2_dev': [m.score(X_valid, y_valid)],
        'oob': [None],
        'n_trees':[m.n_estimators],
        'train_size': [len(y_train)],
        'dev_size': [len(y_valid)],
    }
    if hasattr(m, 'oob_score_'): res['oob'][0] = m.oob_score_
    return pd.DataFrame(res)


# -

m = RandomForestRegressor(n_estimators=10, n_jobs=-1)
# %time m.fit(X_train, y_train)

results = get_scores(m, 'baseline-slow')
results

# An r^2 in the high-80's isn't bad at all (and the RMSLE puts us around rank 100 of 470 on the Kaggle leaderboard), even thought we care overfitting badly by looking at the validation set score.

# ## Speeding things up

nas

df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000, na_dict=nas)
X_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)

nas

m = RandomForestRegressor(n_estimators=10, n_jobs=-1)
# %time m.fit(X_train, y_train)

tmp = get_scores(m, 'speedup')
tmp

results = pd.concat([tmp, results])
results

cols = results.columns[:5]
results[cols].plot.barh(
    x='config', 
    subplots=True, 
    # rot=0, 
    ylim=(0,1), 
    # title=['']*4,
    legend=False,
    figsize=(8,3*results.shape[0])
);

# ## Single tree

m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)

tmp = get_scores(m, 'single tree')
tmp

results = pd.concat([tmp, results])
results

cols = results.columns[:5]
results[cols].plot.barh(
    x='config', 
    subplots=True, 
    # rot=0, 
    ylim=(0,1), 
    # title=['']*4,
    legend=False,
    figsize=(8,3*results.shape[0])
);

# Let's see what happens if we create a bigger tree.

m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)

tmp = get_scores(m, 'single deep tree')
tmp

results = pd.concat([tmp, results])
results

cols = results.columns[:5]
results[cols].plot.barh(
    x='config', 
    subplots=True, 
    # rot=0, 
    ylim=(0,1), 
    # title=['']*4,
    legend=False,
    figsize=(8,3*results.shape[0])
);

# The training set result looks great! But the validation set is worse than our original model. This is why we need to use *bagging* of multiple trees to get more generalizable results.

# ## Bagging

# ### Intro to bagging

# To learn about bagging in random forests, let's start with our basic model again.

m = RandomForestRegressor(n_estimators=10, n_jobs=-1)
m.fit(X_train, y_train)

tmp = get_scores(m, 'baseline-fast')
tmp

results = pd.concat([results, tmp])
results

cols = results.columns[:5]
results[cols].plot.barh(
    x='config', 
    subplots=True, 
    # rot=0, 
    ylim=(0,1), 
    # title=['']*4,
    legend=False,
    figsize=(8,20)
);

# We'll grab the predictions for each individual tree, and look at one example.

preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]

preds.shape

plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);

# The shape of this curve suggests that adding more trees isn't going to help us much. Let's check. (Compare this to our original model on a sample)

m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
get_scores(m, "")

m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
get_scores(m, "")

m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
m.fit(X_train, y_train)

tmp = get_scores(m, "baseline-fast-80")
tmp

results = pd.concat([results, tmp])
results

cols = results.columns[:5]
results[cols].plot.barh(
    x='config', 
    subplots=True, 
    # rot=45,
    ylim=(0,1), 
    # title=['']*4,
    legend=False,
    figsize=(8,20),
);

# ### Out-of-bag (OOB) score

# Is our validation set worse than our training set because we're over-fitting, or because the validation set is for a different time period, or a bit of both? With the existing information we've shown, we can't tell. However, random forests have a very clever trick called *out-of-bag (OOB) error* which can handle this (and more!)
#
# The idea is to calculate error on the training set, but only include the trees in the calculation of a row's error where that row was *not* included in training that tree. This allows us to see whether the model is over-fitting, without needing a separate validation set.
#
# This also has the benefit of allowing us to see whether our model generalizes, even if we only have a small amount of data so want to avoid separating some out to create a validation set.
#
# This is as simple as adding one more parameter to our model constructor. We print the OOB error last in our `print_score` function below.

m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)

tmp = get_scores(m, "baseline-fast-40-oob")
tmp

results = pd.concat([results, tmp])
results

# This shows that our validation set time difference is making an impact, as is model over-fitting.

# ## Reducing over-fitting

# ### Subsampling

# It turns out that one of the easiest ways to avoid over-fitting is also one of the best ways to speed up analysis: *subsampling*. Let's return to using our full dataset, so that we can demonstrate the impact of this technique.

df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)

# The basic idea is this: rather than limit the total amount of data that our model can access, let's instead limit it to a *different* random subset per tree. That way, given enough trees, the model can still see *all* the data, but for each individual tree it'll be just as fast as if we had cut down our dataset as before.

set_rf_samples(20000)

m = RandomForestRegressor(n_estimators=10, n_jobs=-1, oob_score=True)
# %time m.fit(X_train, y_train)

tmp = get_scores(m, "baseline-subsample-10")
tmp

results = pd.concat([results, tmp])
results

cols = results.columns[:5]
results[cols].plot.barh(
    x='config', 
    subplots=True, 
    # rot=45,
    ylim=(0,1), 
    # title=['']*4,
    legend=False,
    figsize=(8,20),
);

# Since each additional tree allows the model to see more data, this approach can make additional trees more useful.

m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)

tmp = get_scores(m, "baseline-subsample-40")
tmp

results = pd.concat([results, tmp])
results

cols = results.columns[:6]
results[cols].plot.barh(
    x='config', 
    subplots=True, 
    # rot=45,
    ylim=(0,1), 
    # title=['']*4,
    legend=False,
    figsize=(8,20),
);

# ### Tree building parameters

# We revert to using a full bootstrap sample in order to show the impact of other over-fitting avoidance methods.

reset_rf_samples()


# Let's get a baseline for this full set to compare to.

def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)

tmp = get_scores(m, 'baseline-slow')
tmp

results = pd.concat([results, tmp])
results

cols = results.columns[:6]
results[cols].plot.barh(
    x='config', 
    subplots=True, 
    # rot=45,
    ylim=(0,1), 
    # title=['']*4,
    legend=False,
    figsize=(8, 20),
);

t=m.estimators_[0].tree_

dectree_max_depth(t)

# Another way to reduce over-fitting is to grow our trees less deeply. We do this by specifying (with `min_samples_leaf`) that we require some minimum number of rows in every leaf node. This has two benefits:
#
# - There are less decision rules for each leaf node; simpler models should generalize better
# - The predictions are made by averaging more rows in the leaf node, resulting in less volatility

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)

tmp = get_scores(m, 'baseline-slow-tuning')
tmp

t=m.estimators_[0].tree_

dectree_max_depth(t)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)

tmp = get_scores(m, 'baseline-slow-tuning')
tmp

# We can also increase the amount of variation amongst the trees by not only use a sample of rows for each tree, but to also using a sample of *columns* for each *split*. We do this by specifying `max_features`, which is the proportion of features to randomly select from at each split.

# - None
# - 0.5
# - 'sqrt'

# - 1, 3, 5, 10, 25, 100

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)

tmp = get_scores(m, 'baseline-slow-tuning')
tmp

results = pd.concat([results, tmp])
results

cols = results.columns[:6]
results[cols].plot.barh(
    x='config', 
    subplots=True, 
    # rot=90,
    # ylim=(0,1), 
    # title=['']*4,
    legend=False,
    figsize=(8,20),
);

# We can't compare our results directly with the Kaggle competition, since it used a different validation set (and we can no longer to submit to this competition) - but we can at least see that we're getting similar results to the winners based on the dataset we have.
#
# The sklearn docs [show an example](http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html) of different `max_features` methods with increasing numbers of trees - as you see, using a subset of features on each split requires using more trees, but results in better models:
# ![sklearn max_features chart](http://scikit-learn.org/stable/_images/sphx_glr_plot_ensemble_oob_001.png)
