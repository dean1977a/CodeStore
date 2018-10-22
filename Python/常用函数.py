# 数据预处理
# 1. 读取数据：
data_macro = pd.read_csv("macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

# 2. 显示为object的属性：
data_train.dtypes[data_train.dtypes=='object']

# 3. 改变数据类型
data_train['material'] = data_train['material'].astype('object')

# 4. 概览数据
data_train.describe(include=['object'])
# 5. 合并两个表（上下）
data_all = pd.concat([data_train, data_test], ignore_index=True)

# 6. 合并两个表（左右）
data_all = pd.merge(data_all, data_macro, on='timestamp', how='left')

# 7. 提取Number， Object特征：
object_columns =  data_all.columns[data_all.dtypes == 'object']
number_columns = data_all.columns[data_all.dtypes != 'object']

# 8. 计算两个特征平均
sa_price = train_df.groupby('sub_area')[['work_share', 'price_doc']].mean()
# 8.1 手工统计信息函数
def stats(x):
    return pd.Series([x.count(),x.min(),x.idxmin(),
    x.quantile(.25),x.median(),
    x.quantile(.75),x.mean(),
    x.max(),x.idxmax(),
    x.mad(),x.var(),
    x.std(),x.skew(),x.kurt()],
    index = ['Count','Min','Whicn_Min',
    'Q1','Median','Q3','Mean',
    'Max','Which_Max','Mad',
    'Var','Std','Skew','Kurt'])
df.apply(stats)

# 数据可视化

# 1. seaborn 画图技巧 
# https://zhuanlan.zhihu.com/p/24464836
plt.figure(figsize=(8, 6))
sns.distplot(a=np.log1p(data_train['price_doc']), bins=50, kde=True)
plt.xlabel("price", fontsize=12)
plt.show()

# 2. 数据中各特征值缺失的个数排序
missing_df = (data_train.isnull().sum(axis=0)/data_train.shape[0]).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df = missing_df.sort_values('missing_count', axis=0,  ascending=True)
width = 0.8
ind = np.arange(missing_df.shape[0])
fig, ax = plt.subplots(figsize=(12, 18))
ax.barh(ind, missing_df['missing_count'], color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df['column_name'], rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()



train_na = (train_df.isnull().sum() / len(train_df)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)

f, ax = plt.subplots(figsize=(12, 8))
plt.xticks(rotation='90')
sns.barplot(x=train_na.index, y=train_na)
ax.set(title='Percent missing data by feature', ylabel='% missing')



# 3. 相关性热图
internal_chars = ['full_sq', 'life_sq', 'floor', 'max_floor', 'build_year', 'num_room', 'kitch_sq', 'state', 'price_doc']
corrmat = train_df[internal_chars].corr()

f, ax = plt.subplots(figsize=(10, 7))
plt.xticks(rotation='90')
sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)


# 4. 散点图
f, ax = plt.subplots(figsize=(10, 7))
ind = train_df[train_df['full_sq'] > 2000].index
plt.scatter(x=train_df.drop(ind)['full_sq'], y=train_df.drop(ind)['price_doc'], c='r', alpha=0.5)
ax.set(title='Price by area in sq meters', xlabel='Area', ylabel='Price')

# 5. 个数图
f, ax = plt.subplots(figsize=(10, 7))
plt.xticks(rotation='90')
sns.countplot(x=train_df['num_room'])
ax.set(title='Distribution of room count', xlabel='num_room')

# 6. 曲线和拟合曲线图
f, ax = plt.subplots(figsize=(12, 6))
by_price = by_df.groupby('build_year')[['build_year', 'price_doc']].mean()
sns.regplot(x="build_year", y="price_doc", data=by_price, scatter=False, order=3, truncate=True)
plt.plot(by_price['build_year'], by_price['price_doc'], color='r')
ax.set(title='Mean price by year of build')

# 7. 小提琴图
f, ax = plt.subplots(figsize=(12, 8))
ind = train_df[train_df['state'].isnull()].index
train_df['price_doc_log10'] = np.log10(train_df['price_doc'])
sns.violinplot(x="state", y="price_doc_log10", data=train_df.drop(ind), inner="box")
# sns.swarmplot(x="state", y="price_doc_log10", data=train_df.dropna(), color="w", alpha=.2);
ax.set(title='Log10 of median price by state of home', xlabel='state', ylabel='log10(price)')

# 8. barplot 
ax = sns.barplot(x="count", y="sub_area", data=sa_vc, orient="h")

# 特征工程
# 1. 移除异常点
ulimit = np.percentile(data_train.price_doc.values, 99)
llimit = np.percentile(data_train.price_doc.values, 1)
data_train.loc[data_train['price_doc'] >ulimit, 'price_doc'] = ulimit
data_train.loc[data_train['price_doc'] <llimit, 'price_doc'] = llimit

# 2. 删除缺失值过半的特征
drop_columns = missing_df.ix[missing_df['missing_count']>0.5, 'column_name'].values
data_train.drop(drop_columns, axis=1, inplace=True)
data_test.drop(drop_columns, axis=1, inplace=True)

# 3. 删除不正常的行数据
data_all.drop(data_train[data_train["life_sq"] > 7000].index, inplace=True)

# 4. 提取时间
# week of year #
data_all["week_of_year"] = data_all["timestamp"].dt.weekofyear
# day of week #
data_all["day_of_week"] = data_all["timestamp"].dt.weekday
# yearmonth
data_all['yearmonth'] = pd.to_datetime(data_all['timestamp'])
data_all['yearmonth'] = data_all['yearmonth'].dt.year*100 + data_all['yearmonth'].dt.month
data_all_groupby = data_all.groupby('yearmonth')

# 5. 连续数据离散化
data_all['floor_25'] = (data_all['floor']>25.0)*1

# 6. 分组来填补平均值
for num in number_columns:
    if(sum(data_all[num].isnull())>0):
        isnull_raw = data_all[num].isnull()
        isnull_yearmonth = data_all.ix[isnull_raw, 'yearmonth'].values
        data_all_groupby[num].transform(lambda x: x.fillna(x.mean()))

# 7. Get_dummies离散化
dummies = pd.get_dummies(data=data_all[ob], prefix="{}#".format(ob))
data_all.drop(ob, axis=1, inplace=True)
data_all = data_all.join(dummies)

# 8. 用radio中位数填补空缺
kitch_ratio = train_df['full_sq']/train_df['kitch_sq']
train_df['kitch_sq']=train_df['kitch_sq'].fillna(train_df['full_sq'] /kitch_ratio.median())

# 9. LabelEncoder 
for ob in object_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(data_train[ob].values))
    data_train[ob] = lbl.fit_transform(list(data_train[ob].values))

# 10. PCA的可视化与转换
from sklearn.decomposition import PCA
components = 20
model = PCA(n_components=components)
model.fit(data_train)
ex_variance = pd.DataFrame({'ex_variance':model.explained_variance_ratio_ [0:components], 'n_component':range(1,components+1)})
ax = sns.barplot(x='n_component', y='ex_variance', data=ex_variance)
ax.set_title('PCA_variance_explained')
plt.show()
data_train = model.fit_transform(data_train)
data_test = model.fit_transform(data_test)

# 创建模型
# 1. import xgboost as xgb
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
train_X = data_train
test_X = data_test
dtrain = xgb.DMatrix(train_X, train_y)
xgb_params={ 'eta': 0.05,
             'max_depth': 5,
             'subsample': 0.7,
             'colsample_bytree': 0.7,
             'objective': 'reg:linear',
             'eval_metric': 'rmse',
             'silent': 1
}

cv_output  = xgb.cv(dict(xgb_params, silent=0), dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=20)
num_boost_round = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=320)
num_boost_round = model.best_iteration
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)
preds = np.exp(model.predict(xgb.DMatrix(test_X, feature_names=test_X.columns.values)))-1
submission = pd.DataFrame()
submission['id'] = test_id
submission["price_doc"]= preds

submission.to_csv("sub.csv",index=False)

# 画feature_importance
%matplotlib inline
fig, ax = plt.subplots(1, 1, figsize=(8, 60))
xgb.plot_importance(model, height=0.5, ax=ax)

# 提取feature_importance
import operator 
importance = model.get_fscore()

df_importance = pd.DataFrame(importance, columns=['feature', 'fscore'])
df_importance.sort_values(ascending=False)

# 其他
# 去除共线性
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer

from statsmodels.stats.outliers_influence import variance_inflation_factor

class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            # Loop repeatedly until we find that all columns within our dataset
            # have a VIF value we're happy with.
            variables = X.columns
            dropped=False
            vif = []
            new_vif = 0
            for var in X.columns:
                new_vif = variance_inflation_factor(X[variables].values, X.columns.get_loc(var))
                vif.append(new_vif)
                if np.isinf(new_vif):
                    break
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                #print 'Dropping {X.columns[{0}]} with vif={{1}}'.format(maxloc, max_vif)
                print X.columns[maxloc]
                print max_vif
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X
transformer = ReduceVIF()
X = data_all
# Only use 10 columns for speed in this example
data_all = transformer.fit_transform(data_train[data_train.columns[0:50]], train_y)

data_all.head()
#  2. Stacking
# Stacking Starter based on Allstate Faron's Script
#https://www.kaggle.com/mmueller/allstate-claims-severity/stacking-starter/run/390867
# Preprocessing from Alexandru Papiu
#https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

SEED = 1
NFOLDS = 3
import pandas as pd
import numpy as np
from scipy.stats import skew
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, Lasso
from math import sqrt

ntrain = data_train.shape[0]
ntest = data_test.shape[0]
print ntrain
print ntest

x_train = np.array(data_train)
x_test = np.array(data_test)
y_train = train_y
kf = KFold(ntrain, n_folds=3, shuffle=True, random_state=SEED)
class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500
}



rd_params={
    'alpha': 10
}


ls_params={
    'alpha': 0.005
}



xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)



xg_oof_train, xg_oof_test = get_oof(xg)
et_oof_train, et_oof_test = get_oof(et)
rf_oof_train, rf_oof_test = get_oof(rf)
rd_oof_train, rd_oof_test = get_oof(rd)
ls_oof_train, ls_oof_test = get_oof(ls)

print("XG-CV: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train))))
print("ET-CV: {}".format(sqrt(mean_squared_error(y_train, et_oof_train))))
print("RF-CV: {}".format(sqrt(mean_squared_error(y_train, rf_oof_train))))
print("RD-CV: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("LS-CV: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))

x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, rd_oof_train, ls_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, rd_oof_test, ls_oof_test), axis=1)

print("{},{}".format(x_train.shape, x_test.shape))


dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)



xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 1,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse'}

res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
