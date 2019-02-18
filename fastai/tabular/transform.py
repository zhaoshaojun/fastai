"Cleaning and feature engineering functions for structured data"
from ..torch_core import *
from pandas.api.types import is_numeric_dtype

__all__ = ['add_datepart', 'cont_cat_split', 'Categorify', 'FillMissing', 'FillStrategy', 'Normalize', 'TabularProc',
           'add_elapsed_times']

def _make_date(df:DataFrame, date_field:str):
    "Make sure `df[field_name]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)

def add_datepart(df:DataFrame, field_name:str, prefix:str=None, drop:bool=True, time:bool=False):
    "Helper function that adds columns relevant to a date in the column `fldname` of `df`."
    _make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower())
    df[prefix + 'Elapsed'] = field.astype(np.int64) // 10 ** 9
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df

def _get_elapsed(df:DataFrame,field_names:Collection[str], date_field:str, base_field:str, prefix:str):
    for f in field_names:
        day1 = np.timedelta64(1, 'D')
        last_date,last_base,res = np.datetime64(),None,[]
        for b,v,d in zip(df[base_field].values, df[f].values, df[date_field].values):
            if last_base is None or b != last_base:
                last_date,last_base = np.datetime64(),b
            if v: last_date = d
            res.append(((d-last_date).astype('timedelta64[D]') / day1))
        df[prefix + f] = res
    return df

def add_elapsed_times(df:DataFrame, field_names:Collection[str], date_field:str, base_field:str):
    field_names = listify(field_names)
    #Make sure date_field is a date and base_field a bool
    df[field_names] = df[field_names].astype('bool')
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype): field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
    
    work_df = df[field_names + [date_field, base_field]]
    work_df = work_df.sort_values([base_field, date_field])
    work_df = _get_elapsed(work_df, field_names, date_field, base_field, 'After')
    work_df = work_df.sort_values([base_field, date_field], ascending=[True, False])
    work_df = _get_elapsed(work_df, field_names, date_field, base_field, 'Before')
    
    for a in ['After' + f for f in field_names] + ['Before' + f for f in field_names]:
        work_df[a] = work_df[a].fillna(0).astype(int)  
    
    for a,s in zip([True, False], ['_bw', '_fw']):
        work_df = work_df.set_index(date_field)
        tmp = (work_df[[base_field] + field_names].sort_index(ascending=a)
                      .groupby(base_field).rolling(7, min_periods=1).sum())
        tmp.drop(base_field,1,inplace=True)
        tmp.reset_index(inplace=True)
        work_df.reset_index(inplace=True)
        work_df = work_df.merge(tmp, 'left', [date_field, base_field], suffixes=['', s])
    work_df.drop(field_names,1,inplace=True)
    return df.merge(work_df, 'left', [date_field, base_field])
        
def cont_cat_split(df, max_card=20, dep_var=None):
    "Helper function that returns column names of cont and cat variables from given df."
    cont_names, cat_names = [], []
    for label in df:
        if label == dep_var: continue
        if len(set(df[label])) > max_card and df[label].dtype == int or df[label].dtype == float: cont_names.append(label)
        else: cat_names.append(label)
    return cont_names, cat_names
        
@dataclass
class TabularProc():
    "A processor for tabular dataframes."
    cat_names:StrList
    cont_names:StrList

    def __call__(self, df:DataFrame, test:bool=False):
        "Apply the correct function to `df` depending on `test`."
        func = self.apply_test if test else self.apply_train
        func(df)

    def apply_train(self, df:DataFrame):
        "Function applied to `df` if it's the train set."
        raise NotImplementedError
    def apply_test(self, df:DataFrame):
        "Function applied to `df` if it's the test set."
        self.apply_train(df)

class Categorify(TabularProc):
    "Transform the categorical variables to that type."
    def apply_train(self, df:DataFrame):
        "Transform `self.cat_names` columns in categorical."
        self.categories = {}
        for n in self.cat_names:
            df.loc[:,n] = df.loc[:,n].astype('category').cat.as_ordered()
            self.categories[n] = df[n].cat.categories

    def apply_test(self, df:DataFrame):
        "Transform `self.cat_names` columns in categorical using the codes decided in `apply_train`."
        for n in self.cat_names:
            df.loc[:,n] = pd.Categorical(df[n], categories=self.categories[n], ordered=True)

FillStrategy = IntEnum('FillStrategy', 'MEDIAN COMMON CONSTANT')

@dataclass
class FillMissing(TabularProc):
    "Fill the missing values in continuous columns."
    fill_strategy:FillStrategy=FillStrategy.MEDIAN
    add_col:bool=True
    fill_val:float=0.
    def apply_train(self, df:DataFrame):
        "Fill missing values in `self.cont_names` according to `self.fill_strategy`."
        self.na_dict = {}
        for name in self.cont_names:
            if pd.isnull(df.loc[:,name]).sum():
                if self.add_col:
                    df.loc[:,name+'_na'] = pd.isnull(df.loc[:,name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                if self.fill_strategy == FillStrategy.MEDIAN: filler = df.loc[:,name].median()
                elif self.fill_strategy == FillStrategy.CONSTANT: filler = self.fill_val
                else: filler = df.loc[:,name].dropna().value_counts().idxmax()
                df.loc[:,name] = df.loc[:,name].fillna(filler)
                self.na_dict[name] = filler

    def apply_test(self, df:DataFrame):
        "Fill missing values in `self.cont_names` like in `apply_train`."
        for name in self.cont_names:
            if name in self.na_dict:
                if self.add_col:
                    df.loc[:,name+'_na'] = pd.isnull(df[name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                df.loc[:,name] = df.loc[:,name].fillna(self.na_dict[name])
            elif pd.isnull(df[name]).sum() != 0:
                raise Exception(f"""There are nan values in field {name} but there were none in the training set. 
                Please fix those manually.""")

class Normalize(TabularProc):
    "Normalize the continuous variables."
    def apply_train(self, df:DataFrame):
        "Comput the means and stds of `self.cont_names` columns to normalize them."
        self.means,self.stds = {},{}
        for n in self.cont_names:
            assert is_numeric_dtype(df[n]), (f"""Cannot normalize '{n}' column as it isn't numerical.
                Are you sure it doesn't belong in the categorical set of columns?""")
            self.means[n],self.stds[n] = df.loc[:,n].mean(),df.loc[:,n].std()
            df.loc[:,n] = (df.loc[:,n]-self.means[n]) / (1e-7 + self.stds[n])

    def apply_test(self, df:DataFrame):
        "Normalize `self.cont_names` with the same statistics as in `apply_train`."
        for n in self.cont_names:
            df.loc[:,n] = (df.loc[:,n]-self.means[n]) / (1e-7 + self.stds[n])
