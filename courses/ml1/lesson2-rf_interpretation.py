# **Important: This notebook will only work with fastai-0.7.x. Do not try to run any fastai-1.x code from this path in the repository because it will load fastai-0.7.x**

# # Random Forest Model interpretation

# %load_ext autoreload
# %autoreload 2

# +
# %matplotlib inline

from fastai.imports import *
from fastai.structured import *
# from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
# -

set_plot_sizes(12,14,16)

# ## Load in our data from last lesson

# +
PATH = "data/bulldozers/"

df_raw = pd.read_feather('tmp/bulldozers-raw')
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')


# -

def split_vals(a,n): return a[:n], a[n:]
n_valid = 12000
n_trn = len(df_trn)-n_valid
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)


# +
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def get_scores(m, config=None):
    res = {
        'config': [config],
        'rmse_train': [rmse(m.predict(X_train), y_train)],
        'rmse_dev': [rmse(m.predict(X_valid), y_valid)],
        'r2_train': [m.score(X_train, y_train)],
        'r2_dev': [m.score(X_valid, y_valid)],
        'r2_oob': [None],
        'n_trees':[m.n_estimators],
        'train_size': [len(y_train)],
        'dev_size': [len(y_valid)],
    }
    if hasattr(m, 'oob_score_'): res['r2_oob'][0] = m.oob_score_
    return pd.DataFrame(res)


# -

df_raw

# # Confidence based on tree variance

# For model interpretation, there's no need to use the full dataset on each tree - using a subset will be both faster, and also provide better interpretability (since an overfit model will not provide much variance across trees).

set_rf_samples(50000)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)

results = get_scores(m, 'baseline-subsample-tuning')
results

# We saw how the model averages predictions across the trees to get an estimate - but how can we know the confidence of the estimate? One simple way is to use the standard deviation of predictions, instead of just the mean. This tells us the *relative* confidence of predictions - that is, for rows where the trees give very different results, you would want to be more cautious of using those results, compared to cases where they are more consistent. Using the same example as in the last lesson when we looked at bagging:

# %time preds = np.stack([t.predict(X_valid) for t in m.estimators_])
np.mean(preds[:,0]), np.std(preds[:,0])


# When we use python to loop through trees like this, we're calculating each in series, which is slow! We can use parallel processing to speed things up:

def get_preds(t): return t.predict(X_valid)
# %time preds = np.stack(parallel_trees(m, get_preds))
np.mean(preds[:,0]), np.std(preds[:,0])

# We can see that different trees are giving different estimates this this auction. In order to see how prediction confidence varies, we can add this into our dataset.

x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)
x.Enclosure.value_counts().plot.barh();

flds = ['Enclosure', 'SalePrice', 'pred', 'pred_std']
enc_summ = x[flds].groupby('Enclosure', as_index=False).mean()
enc_summ

enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]
enc_summ.plot('Enclosure', 'SalePrice', 'barh', xlim=(0,11));

enc_summ.plot('Enclosure', 'pred', 'barh', xerr='pred_std', alpha=0.6, xlim=(0,11));

# *Question*: Why are the predictions nearly exactly right, but the error bars are quite wide?

raw_valid.ProductSize.value_counts().plot.barh();

flds = ['ProductSize', 'SalePrice', 'pred', 'pred_std']
summ = x[flds].groupby(flds[0]).mean()
summ

(summ.pred_std/summ.pred).sort_values(ascending=False)

# # Feature importance

# It's not normally enough to just to know that a model can make accurate predictions - we also want to know *how* it's making predictions. The most important way to see this is with *feature importance*.

fi = rf_feat_importance(m, df_trn); fi[:10]

fi.plot('cols', 'imp', figsize=(10,6), legend=False);


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


plot_fi(fi[:30]);

to_keep = fi[fi.imp>0.005].cols; len(to_keep)

df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5,
                          n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)

tmp = get_scores(m, 'fi')
tmp

results = pd.concat([tmp, results])
results[::-1]

fi = rf_feat_importance(m, df_keep)
plot_fi(fi);

# ## One-hot encoding

# proc_df's optional *max_n_cat* argument will turn some categorical variables into new columns.
#
# For example, the column **ProductSize** which has 6 categories:
#
# * Large
# * Large / Medium
# * Medium
# * Compact
# * Small
# * Mini
#
# gets turned into 6 new columns:
#
# * ProductSize_Large
# * ProductSize_Large / Medium
# * ProductSize_Medium
# * ProductSize_Compact
# * ProductSize_Small
# * ProductSize_Mini
#
# and the column **ProductSize** gets removed.
#
# It will only happen to columns whose number of categories is no bigger than the value of the *max_n_cat* argument.
#
# Now some of these new columns may prove to have more important features than in the earlier situation, where all categories were in one column.

# +
df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
# -

tmp = get_scores(m, 'one-hot')
tmp

results = pd.concat([tmp, results])
results[::-1]

fi = rf_feat_importance(m, df_trn2)
plot_fi(fi[:25]);

# # Removing redundant features

# One thing that makes this harder to interpret is that there seem to be some variables with very similar meanings. Let's try to remove redundent features.

from scipy.cluster import hierarchy as hc

corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()

sorted(list(df_keep.columns))

len(corr), len(corr[0])


# Let's try removing some of these related features to see if the model can be simplified without impacting the accuracy.

def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, max_features=0.6, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_


# Here's our baseline.

get_oob(df_keep)

# Now we try removing each variable one at a time.

for c in ('saleYear', 'saleElapsed', 'fiModelDesc', 'fiBaseModel', 'Grouser_Tracks', 'Coupler_System'):
    print(c, get_oob(df_keep.drop(c, axis=1)))

# It looks like we can try one from each group for removal. Let's see what that does.

to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_oob(df_keep.drop(to_drop, axis=1))

# Looking good! Let's use this dataframe from here. We'll save the list of columns so we can reuse it later.

df_keep.drop(to_drop, axis=1, inplace=True)
X_train, X_valid = split_vals(df_keep, n_trn)

# +
# np.save('tmp/keep_cols.npy', np.array(df_keep.columns))
# -

# keep_cols = np.load('tmp/keep_cols.npy')
# df_keep = df_trn[keep_cols]
keep_cols = df_keep.columns

sorted(list(keep_cols))

# And let's see how this model looks on the full dataset.

reset_rf_samples()

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)

tmp = get_scores(m, 'baseline-slow-dedup')
tmp

results = pd.concat([tmp, results])
results[::-1]

# # Partial dependence

from pdpbox import pdp
from plotnine import *

set_rf_samples(50000)

# This next analysis will be a little easier if we use the 1-hot encoded categorical variables, so let's load them up again.

df_trn2, y_trn, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train);

plot_fi(rf_feat_importance(m, df_trn2)[:10]);

df_raw.plot('YearMade', 'saleElapsed', 'scatter', alpha=0.01, figsize=(10,8));

x_all = get_sample(df_raw[df_raw.YearMade>1930], 500)

ggplot(x_all, aes('YearMade', 'SalePrice'))+stat_smooth(se=True, method='loess')

x = get_sample(X_train[X_train.YearMade>1930], 500)


def plot_pdp_old(feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                        cluster=clusters is not None,
                        n_cluster_centers=clusters)


def plot_pdp(feat, clusters = None, feat_name = None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, feature = feat, model_features = x.columns)
    return pdp.pdp_plot(p, feat_name, plot_lines = True,
                        cluster = clusters is not None,
                        n_cluster_centers = clusters)


plot_pdp('YearMade')

plot_pdp('YearMade', clusters=5)

sorted(list(x.columns))

try:
    feats = ['saleElapsed', 'YearMade']
    p = pdp.pdp_interact(m, x, x.columns, feats)
    pdp.pdp_interact_plot(p, feats)
except:
    print("ignore errors")

plot_pdp(['Enclosure_EROPS w AC', 'Enclosure_EROPS', 'Enclosure_OROPS'], 5, 'Enclosure')

df_raw.YearMade[df_raw.YearMade<1950] = 1950
df_keep['age'] = df_raw['age'] = df_raw.saleYear-df_raw.YearMade

X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
plot_fi(rf_feat_importance(m, df_keep));

# # Tree interpreter

from treeinterpreter import treeinterpreter as ti

df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)

row = X_valid.values[None,0]; row

prediction, bias, contributions = ti.predict(m, row)

prediction[0], bias[0]

idxs = np.argsort(contributions[0])

pd.concat(
    [pd.DataFrame(contributions[0]), 
     pd.DataFrame(contributions[0][idxs]),
     pd.DataFrame(idxs)], 
    axis=1
)

[o for o in zip(df_keep.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]

contributions[0].sum(), bias[0], contributions[0].sum() + bias[0], prediction[0]

# # Extrapolation

df_ext = df_keep.copy()
df_ext['is_valid'] = 1
df_ext.is_valid[:n_trn] = 0
x, y, nas = proc_df(df_ext, 'is_valid')

m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_

fi = rf_feat_importance(m, x); fi[:10]

feats=['SalesID', 'saleElapsed', 'MachineID']

(X_train[feats]/1000).describe()

(X_valid[feats]/1000).describe()

x.drop(feats, axis=1, inplace=True)

m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_

fi = rf_feat_importance(m, x); fi[:10]

feats=['age', 'YearMade', 'saleDayofyear']

(X_train[feats]).describe()

(X_valid[feats]).describe()

set_rf_samples(50000)

feats=['SalesID', 'saleElapsed', 'MachineID', 
       'age', 'YearMade', 'saleDayofyear']

X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
get_scores(m, '')

for f in feats:
    df_subs = df_keep.drop(f, axis=1)
    X_train, X_valid = split_vals(df_subs, n_trn)
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    print(f)
    display(get_scores(m, ''))

# +
# reset_rf_samples()
# -

df_subs = df_keep.drop(['SalesID', 'MachineID', 'saleDayofyear'], axis=1)
X_train, X_valid = split_vals(df_subs, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
get_scores(m, '')

reset_rf_samples()

df_subs = df_keep.drop(['SalesID', 'MachineID', 'saleDayofyear'], axis=1)
X_train, X_valid = split_vals(df_subs, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
get_scores(m, '')

plot_fi(rf_feat_importance(m, X_train));

np.save('tmp/subs_cols.npy', np.array(df_subs.columns))

# # Our final model!

m = RandomForestRegressor(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)
# %time m.fit(X_train, y_train)
get_scores(m, "final")

tmp = get_scores(m, "final")
tmp

results

results = pd.concat([tmp, results])
results[::-1]

cols = results.columns[:6]
results[cols].plot.barh(
    x='config',
    subplots=True,
    # rot=90,
    # ylim=(0,1),
    # title=['']*4,
    legend=False,
    figsize=(8,3*results.shape[0])
);


