# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:18:09 2017

@author: Hank Kuang
@mission: Credit Risk Score Calculate
"""


import pandas as pd
from pandas import Series, DataFrame
import numpy as np


class CreditScoreCalculater:
    
    
    def __init__(self, credit_card, datasets):
        """
        Params
        ------------------------------------------
        credit_card: pandas dataframe
        datasets: pandas dataframe
        """
        self.credit_card = credit_card
        self.datasets = datasets
    
    
    def ScoresOutput(self):
        """
        Func:
        output the results
        -------------------------
        Return
        pandas Series, index is dataset's index, values is scores
        """
        scores_list = []
        
        # 解析所有评分卡，将评分卡转为可计算的表格形式
        # 连续型变量包括：scores, lower&upper
        # 字符型变量包括：scores，valuelist
        scoresMap = dict()
        weights = dict()
        
        not_in_credit_cards = []
        for v_name in self.datasets.columns:
            if v_name in self.credit_card.index.levels[0]:
                
                weights[v_name], scoresMap[v_name] = self.ParseCreditCard(v_name)
            else:
                not_in_credit_cards.append(v_name)
        # parse
        for i in self.datasets.index:
            
            p = self.datasets.loc[i,:]
            # 计算个人每个变量下对应值得分数
            scores = [self.calScore(i, v, weights[i], scoresMap[i]) 
                     for i, v in zip(p.index, p.values) 
                     if i not in not_in_credit_cards]
            # 加总评分
            scoreSum = sum(scores)
            scores_list.append(scoreSum)
            
        score_series = Series(scores_list, index=self.datasets.index)
        score_series.name = 'credit_score'
        # 如果评分卡包含基准分，则加入基准分
        if 'baseScore' in self.credit_card.index.levels[0]:
            baseScores = self.credit_card.loc['baseScore']['score'][0]
            w = self.credit_card.loc['baseScore', :].index.levels[0][0]
            baseScore = float(baseScores) * float(w)
            score_series = score_series + baseScore
        
        return score_series
    
    
    def ParseCreditCard(self, varname):
        """
        Func:
        Parse CreditCard to table, 
        contains scores, rule, and weights  
        ----------------------------------------
        Params
        varname: variable name
        ----------------------------------------
        Return
        3 kinds of table, single variable's weight
        """
        # 根据变量名提取子评分卡，并将子评分卡索引进行转换，提取权重
        sub_card = self.credit_card.loc[varname, :]
        weight = sub_card.index.levels[0][0]
        sub_card.index = range(1, len(sub_card)+1)
        
        try:
            if '--' in sub_card['range'].iloc[0]:
            
                table = self.rangeParse(sub_card, method=1)
            
            elif ';' in sub_card['range'].iloc[0]:
                table = self.rangeParse(sub_card, method=2)
            
            else:
                table = self.rangeParse(sub_card, method=3)
        
        except TypeError:
            table = self.rangeParse(sub_card, method=3)
        return weight, table
    
       
    def calScore(self, varname, value, w, scoreMap):
        """
        Func
        calculate score of single variable
        -------------------------------------------
        Params
        varname:
        value:
        w:
        scoreMap:
        -------------------------------------------
        score 
        """
        
        if 'lower' in scoreMap.columns:
            for i in scoreMap.index:
                if value >= scoreMap.loc[i, 'lower'] and value < scoreMap.loc[i, 'upper']:
                    score = scoreMap.loc[i, 'score']
                else:
                    pass
        
        else:
            for i in scoreMap.index:
                value = str(value)
                try:
                    
                    if value in scoreMap.loc[i, 'range']:
                        score = scoreMap.loc[i, 'score']
                except TypeError:
                    
                    if value == scoreMap.loc[i, 'range']:
                        score = scoreMap.loc[i, 'score']
        
        score = score * w
        
        return score 
    
    def rangeParse(self, sub_card, method):
        """
        Func
        Parse range to cal rule
        -------------------------------------
        sub_card:
        method:
        -------------------------------------
        Return
        new sub_card
        """
        if method == 1:
            lowerList = []
            upperList = []
            for item in sub_card['range']:
                lower, upper = self.Parse(item)
                lowerList.append(lower)
                upperList.append(upper)
            sub_card['lower'] = lowerList
            sub_card['upper'] = upperList
            sub_card['lower'] = sub_card['lower'].astype(np.float_)
            sub_card['upper'] = sub_card['upper'].astype(np.float_)
        elif method == 2:
            lst = []
            for item in sub_card['range']:
                l = item.split(';')
                lst.append(l)
                lst = [str(l) for l in lst]
        
            sub_card.loc[:, 'range'] = lst
        
        else:
            try:
                sub_card.loc[:, 'range'] = [str(s) for s in sub_card.loc[:, 'range']]
            except Exception:
                print("something different in sub_card")
                print(sub_card)
        return sub_card
            
    
    def Parse(self, ranges_):
        """
        Func
        parse lower&upper
        ---------------------------
        Params
        ranges_: ranges of every rows
        -----------------------------
        Return
        upper & lower
        """
        lst = ranges_.split('--')
        return lst[0], lst[1]
        





def calculateCreditScore(datasets, credit_card):
    """
    Func
    calculate scores for every rows in datasets
    -----------------------------------------------
    Params
    datasets: pandas dataframe
    credit_card: pandas dataframe, scorescard
    -----------------------------------------------
    Return
    res: pandas series, 
    """
    score_cal = CreditScoreCalculater(credit_card, datasets)
    res = score_cal.ScoresOutput()
    return res
    
"""
example:

import os
import pandas as pd

os.chdir("E:/anaylsis_engine/creditCardArtcle/codes")
credit_card = pd.read_excel("credit_card.xlsx")

credit_card['varname'] = credit_card['varname'].fillna(method='ffill')
credit_card['weights'] = credit_card['weights'].fillna(method='ffill')
credit_card['binCode'] = credit_card['binCode'].fillna(method='ffill')

credit_card.index = credit_card[['varname', 'weights', 'binCode']]
credit_card = pd.DataFrame(credit_card[['score', 'range']], 
                           index=[credit_card['varname'], credit_card['weights'], credit_card['binCode']])
    

os.chdir("D:/dowload/Chrome_download/default-of-credit-card-clients-dataset")
data = pd.read_csv("creditCard_UCI.csv")

score_cal = CreditScoreCalculater(credit_card, data)
res = score_cal.ScoresOutput()

"""






# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:28:07 2018

@author: Administrator
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import copy

## %matplotlib inline

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20

train = pd.read_csv(u'C:/Users/Administrator/Desktop/cs-training.csv')

## 数值变量
numerical = ['RevolvingUtilizationOfUnsecuredLines',
             'age',
             'NumberOfTime30-59DaysPastDueNotWorse',
             'DebtRatio',
             'MonthlyIncome',
             'NumberOfOpenCreditLinesAndLoans',
             'NumberOfTimes90DaysLate',
             'NumberRealEstateLoansOrLines',
             'NumberOfTime60-89DaysPastDueNotWorse',
             'NumberOfDependents']

target_var = ['SeriousDlqin2yrs']

train_X = train[numerical]
train_Y = train[target_var]


## 对变量进行缺失值分析
missing = train_X.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

## 只有两列变量有值，本方法中不对缺失值做填充处理，用区别于原始分布的数值代替
train_X['NumberOfDependents'].fillna(-999, inplace=True)
train_X['MonthlyIncome'].fillna(-999, inplace = True)

## 观察异常值
    
sns.boxplot(train['RevolvingUtilizationOfUnsecuredLines'])
sns.boxplot(train['age'])
sns.boxplot(train['NumberOfTime30-59DaysPastDueNotWorse'])
sns.boxplot(train['DebtRatio'])
sns.boxplot(train['MonthlyIncome'])
sns.boxplot(train['NumberOfOpenCreditLinesAndLoans'])
sns.boxplot(train['NumberOfTimes90DaysLate'])
sns.boxplot(train['NumberRealEstateLoansOrLines'])
sns.boxplot(train['NumberOfTime60-89DaysPastDueNotWorse'])
sns.boxplot(train['NumberOfDependents'])

## 箱型图识别异常值标准
outile_var = ['RevolvingUtilizationOfUnsecuredLines',
              'NumberOfTime30-59DaysPastDueNotWorse',
              'DebtRatio',
              'MonthlyIncome',
              'NumberOfOpenCreditLinesAndLoans',
              'NumberOfTimes90DaysLate',
              'NumberRealEstateLoansOrLines',
              'NumberOfTime60-89DaysPastDueNotWorse',
              'NumberOfDependents']


### 用投票法来决定是否为异常值
for col in outile_var:
    ##1
    qua_U = train[col].quantile(0.75)
    qua_L = train[col].quantile(0.25)
    IQR = qua_U - qua_L
    
    #2
    mean_values =  train[col].mean()
    std_values = train[col].std()
     
    #3
    qua_U1 = train[col].quantile(0.95)
    
    outile1 = qua_U + 1.5 * IQR
    outile2 = qua_U1
    outile3 = mean_values + 3 * std_values
    
    median = train[col].median()
    train_X.loc[((train_X[col] > outile1) & (train_X[col] > outile2) & (train_X[col] > outile3)), col] = median
    
    

### 对特征做log 转化

def log_transform(feature):
    train_X[feature + '_log'] = np.log1p(train_X[feature].values)

## 对特征进行二次转化
def quadratic(feature):
    train_X[feature + '_2'] = train_X[feature]**2


log_transform('RevolvingUtilizationOfUnsecuredLines')
log_transform('NumberOfTime30-59DaysPastDueNotWorse')
log_transform('age')
log_transform('DebtRatio')
log_transform('MonthlyIncome')
log_transform('NumberOfOpenCreditLinesAndLoans')
log_transform('NumberOfTimes90DaysLate')
log_transform('NumberRealEstateLoansOrLines')
log_transform('NumberOfDependents')
log_transform('NumberOfTime60-89DaysPastDueNotWorse')


quadratic('RevolvingUtilizationOfUnsecuredLines')
quadratic('NumberOfTime30-59DaysPastDueNotWorse')
quadratic('age')
quadratic('DebtRatio')
quadratic('MonthlyIncome')
quadratic('NumberOfOpenCreditLinesAndLoans')
quadratic('NumberOfTimes90DaysLate')
quadratic('NumberRealEstateLoansOrLines')
quadratic('NumberOfDependents')
quadratic('NumberOfTime60-89DaysPastDueNotWorse')

## 对特征进行组合
## 月负债
train_X['month_debt'] = train_X['DebtRatio'] * train_X['MonthlyIncome']

## 人均收入
#train_X['income_person'] = train_X['NumberOfDependents'] / train_X['MonthlyIncome']

train_X.fillna(-999, inplace=True)

## 所有的数值特征进行特征分箱处理
## 离散特征，如果因子数过多，也可采用这种方式进行因子合并

## 本文采用卡方分箱，自底向上的数据离散化方法，具有最小卡方值的相领区间合并一起，直到满足确定的停止准则
## pandas自带三种分箱方式
# 卡方分箱
'''
from reportgen.utils import Discretization
dis=Discretization(method='chimerge',max_intervals=20)
dis.fit_transform(train_X, train_Y)
'''

## 自写卡方最优分箱过程
def get_chi2(X, col):
    '''
    计算卡方统计量
    '''
    # 计算样本期望频率
    
    pos_cnt = X['SeriousDlqin2yrs'].sum()
    all_cnt = X['SeriousDlqin2yrs'].count()
    expected_ratio = float(pos_cnt) / all_cnt 
    
    # 对变量按属性值从大到小排序
    df = X[[col, 'SeriousDlqin2yrs']]
    col_value = list(set(df[col]))
    col_value.sort()
    
    # 计算每一个区间的卡方统计量
    
    chi_list = []
    pos_list = []
    expected_pos_list = []
    
    for value in col_value:
        df_pos_cnt = df.loc[df[col] == value, 'SeriousDlqin2yrs'].sum()
        df_all_cnt = df.loc[df[col] == value,'SeriousDlqin2yrs'].count()
        
        expected_pos_cnt = df_all_cnt * expected_ratio
        chi_square = (df_pos_cnt - expected_pos_cnt)**2 / expected_pos_cnt
        chi_list.append(chi_square)
        pos_list.append(df_pos_cnt)
        expected_pos_list.append(expected_pos_cnt)
    
    # 导出结果到dataframe
    chi_result = pd.DataFrame({col: col_value, 'chi_square':chi_list,
                               'pos_cnt':pos_list, 'expected_pos_cnt':expected_pos_list})
    return chi_result

def chiMerge(chi_result, maxInterval=5):
       
    '''
    根据最大区间数限制法则，进行区间合并
    '''
    
    group_cnt = len(chi_result)
    # 如果变量区间超过最大分箱限制，则根据合并原则进行合并，直至在maxInterval之内
    
    while(group_cnt > maxInterval):
        
        ## 取出卡方值最小的区间
        min_index = chi_result[chi_result['chi_square'] == chi_result['chi_square'].min()].index.tolist()[0]
        
        # 如果分箱区间在最前,则向下合并
        if min_index == 0:
            chi_result = merge_chiSquare(chi_result, min_index+1, min_index)
        
        # 如果分箱区间在最后，则向上合并
        elif min_index == group_cnt-1:
            chi_result = merge_chiSquare(chi_result, min_index-1, min_index)
        
        # 如果分箱区间在中间，则判断两边的卡方值，选择最小卡方进行合并
        else:
            if chi_result.loc[min_index-1, 'chi_square'] > chi_result.loc[min_index+1, 'chi_square']:
                chi_result = merge_chiSquare(chi_result, min_index, min_index+1)
            else:
                chi_result = merge_chiSquare(chi_result, min_index-1, min_index)
        
        group_cnt = len(chi_result)
    
    return chi_result


def merge_chiSquare(chi_result, index, mergeIndex, a = 'expected_pos_cnt',
                    b = 'pos_cnt', c = 'chi_square'):
    '''
    按index进行合并，并计算合并后的卡方值
    mergeindex 是合并后的序列值
    
    '''
    chi_result.loc[mergeIndex, a] = chi_result.loc[mergeIndex, a] + chi_result.loc[index, a]
    chi_result.loc[mergeIndex, b] = chi_result.loc[mergeIndex, b] + chi_result.loc[index, b]
    ## 两个区间合并后，新的chi2值如何计算
    chi_result.loc[mergeIndex, c] = (chi_result.loc[mergeIndex, b] - chi_result.loc[mergeIndex, a])**2 /chi_result.loc[mergeIndex, a]
    
    chi_result = chi_result.drop([index])
    
    ## 重置index
    chi_result = chi_result.reset_index(drop=True)
    
    return chi_result

## chi2分箱主流程
# 1：计算初始chi2 result
## 合并X数据集与Y数据集

### 先对数据进行等频分箱，提高卡方分箱的效率

## 注意对原始数据的拷贝
freq_train_X = copy.deepcopy(train_X)

def get_freq(train_X, col, bind):
    col_data = train_X[col]
    col_data_sort = col_data.sort_values().reset_index(drop=True)
    col_data_cnt = col_data.count()
    length = col_data_cnt / bind
    col_index = np.append(np.arange(length, col_data_cnt, length), (col_data_cnt - 1))
    col_interval = list(set(col_data_sort[col_index]))
    return col_interval    
    
    
for col in train_X.columns:
    print "start get " + col + " 等频 result"
    col_interval = get_freq(train_X, col, 200)
    col_interval.sort()
    for i, val in enumerate(col_interval):
        if i == 0:
            freq_train_X.loc[train_X[col] <= val, col] = i + 1 
            
        else:
            freq_train_X.loc[(train_X[col]<= val) & (train_X[col] > col_interval[i-1]), col] = i + 1
        
    
    
    
train_all = pd.concat([freq_train_X, train_Y], axis=1)
chi_result_all = dict()

for col in freq_train_X.columns:
    print "start get " + col + " chi2 result"
    chi2_result = get_chi2(train_all, col)
    chi2_merge = chiMerge(chi2_result, maxInterval = 10)
    
    
    chi_result_all[col] = chi2_merge




## 2:进行区间合并
##chi2_merge = chiMerge(chi2_result, maxInterval = 10)
    
# 等距分箱
## pd.cut(x,bins=10)
# 等频分箱
## pd.qcut(x,bins=10)
    

### 进行WOE编码

woe_iv={} ### 计算特征的IV值

def get_woevalue(train_all, col, chi2_merge):
    ## 计算所有样本中，响应客户和未响应客户的比例
    df_pos_cnt = train_all['SeriousDlqin2yrs'].sum()
    df_neg_cnt = train_all['SeriousDlqin2yrs'].count() - df_pos_cnt
    
    df_ratio = df_pos_cnt / (df_neg_cnt * 1.0)
    
        
    col_interval = chi2_merge[col].values
    woe_list = []
    iv_list = []
    
    for i, val in enumerate(col_interval):
        if i == 0:
            col_pos_cnt = train_all.loc[train_all[col]<= val, 'SeriousDlqin2yrs'].sum()
            col_all_cnt = train_all.loc[train_all[col]<= val, 'SeriousDlqin2yrs'].count()
            col_neg_cnt = col_all_cnt - col_pos_cnt
        
        else:
            col_pos_cnt = train_all.loc[(train_all[col]<= val) & (train_all[col] > col_interval[i-1]), 'SeriousDlqin2yrs'].sum()
            col_all_cnt = train_all.loc[(train_all[col]<= val) & (train_all[col] > col_interval[i-1]), 'SeriousDlqin2yrs'].count()
            col_neg_cnt = col_all_cnt - col_pos_cnt
        
        if col_neg_cnt == 0:
            col_neg_cnt = col_neg_cnt + 1
        
        col_ratio = col_pos_cnt / (col_neg_cnt * 1.0)
        
        
        woei = np.log(col_ratio / df_ratio)
        ivi = woei * ((col_pos_cnt / (df_pos_cnt * 1.0)) - (col_neg_cnt / (df_neg_cnt * 1.0)))
        woe_list.append(woei)
        iv_list.append(ivi)
    
    IV = sum(iv_list)
    
    return woe_list, IV
        
        
for col in freq_train_X.columns:
    
    ## 首先对特征进行分箱转化
    chi2_merge = chi_result_all[col]
    woe_list, iv = get_woevalue(train_all, col, chi2_merge)
    woe_iv[col] = {'woe_list': woe_list, 'iv': iv}
    
### 根据计算的IV值进行特征筛选
for col in freq_train_X.columns:
    iv = woe_iv[col]['iv']
    if iv < 0.02:
        freq_train_X.drop([col], axis=1) ## 删除IV值过小的特征

### 对留下的特征进行WOE编码转化,WOE编码只是为了使得评分卡的格式更加标准化，并不能提高模型的效果，分箱完过后，直接建立模型，一样可以达到目的

woe_train_X = copy.deepcopy(freq_train_X)

for col in freq_train_X.columns:
    woe_list = woe_iv[col]['woe_list']
    col_interval = chi_result_all[col][col].values
    
    for i, val in enumerate(col_interval):
        if i == 0:
            woe_train_X.loc[freq_train_X[col] <= val, col] = woe_list[i]
        else:
            woe_train_X.loc[(freq_train_X[col] <= val) & (freq_train_X[col] > col_interval[i-1]), col] = woe_list[i]
        


####对最终的数据集进行建模
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression 
from sklearn.metrics import auc,roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score

### 切分训练集和测试集
X_train, X_test,y_train,y_test = train_test_split(woe_train_X, train_Y, test_size=0.3)

c, r = y_train.shape
y_train = y_train.values.reshape(c,)

## 建立logistic回归模型 
lr = LogisticRegression(C=0.01)
lr.fit(X_train, y_train)

## 用拟合好的模型预测训练集
y_train_proba = lr.predict_proba(X_train)
y_train_label = lr.predict(X_train)

## 用拟合好的模型预测测试集
y_test_proba = lr.predict_proba(X_test)
y_test_label = lr.predict(X_test)


print('训练集准确率：{:.2%}'.format(accuracy_score(y_train, y_train_label)))
print('测试集准确率：{:.2%}'.format(accuracy_score(y_test, y_test_label)))

print('训练集精度：{:.2%}'.format(precision_score(y_train, y_train_label)))
print('测试集精度：{:.2%}'.format(precision_score(y_test, y_test_label)))

print('训练集召回率：{:.2%}'.format(recall_score(y_train, y_train_label)))
print('测试集召回率：{:.2%}'.format(recall_score(y_test, y_test_label)))

print('训练集AUC：{:.2%}'.format(roc_auc_score(y_train, y_train_proba[:,1])))
print('测试集AUC：{:.2%}'.format(roc_auc_score(y_test, y_test_proba[:,1])))
# ROC曲线和KS统计量
### ROC反映的是 TPR 与 FPR 之间的关系
### TPR = TP / (TP + FN) 灵敏度
### FPR = FP / (TN + FP) 误警率
### 绘制的是在不同阈值下的两者关系
### KS值小于0.2认为模型无鉴别能力

fpr, tpr, thresholds = roc_curve(y_test,y_test_proba[:,1], pos_label=1)
auc_score = auc(fpr,tpr)
w = tpr - fpr
ks_score = w.max()
ks_x = fpr[w.argmax()]
ks_y = tpr[w.argmax()]
fig,ax = plt.subplots()
ax.plot(fpr,tpr,label='AUC=%.5f'%auc_score)
ax.set_title('Receiver Operating Characteristic')
ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
ax.plot([ks_x,ks_x], [ks_x,ks_y], '--', color='red')
ax.text(ks_x,(ks_x+ks_y)/2,'  KS=%.5f'%ks_score)
ax.legend()
fig.show()          


   

### 模型参数寻优
## 采用网格搜索法，寻找最优参数  
from sklearn.grid_search import GridSearchCV  
from sklearn.pipeline import Pipeline  
 

pipeline = Pipeline([  
('clf', LogisticRegression())  
])  
parameters = {  
'clf__penalty': ('l1', 'l2'),  
'clf__C': (0.01,0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1, 10)}
  
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='roc_auc', cv=3) 
#c, r = y_train.shape
#y_train = y_train.values.reshape(c,)

if __name__ == '__main__':
    grid_search.fit(X_train, y_train)
  
print('最佳效果：%0.3f' % grid_search.best_score_)  
print('最优参数组合：')  
best_parameters = grid_search.best_estimator_.get_params()  
for param_name in sorted(parameters.keys()):  
    print('\t%s: %r' % (param_name, best_parameters[param_name]))  

predictions = grid_search.predict(X_test)  
print('准确率：', accuracy_score(y_test, predictions))  
print('精确率：', precision_score(y_test, predictions))  
print('召回率：', recall_score(y_test, predictions))

### 采用其他模型进行训练，评估效果

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier


## 建立不同的分类器模型 
model = AdaBoostClassifier()

model.fit(X_train, y_train)

## 用拟合好的模型预测训练集
y_train_proba = model.predict_proba(X_train)
y_train_label = model.predict(X_train)

## 用拟合好的模型预测测试集
y_test_proba = model.predict_proba(X_test)
y_test_label = model.predict(X_test)


print('训练集准确率：{:.2%}'.format(accuracy_score(y_train, y_train_label)))
print('测试集准确率：{:.2%}'.format(accuracy_score(y_test, y_test_label)))

print('训练集精度：{:.2%}'.format(precision_score(y_train, y_train_label)))
print('测试集精度：{:.2%}'.format(precision_score(y_test, y_test_label)))

print('训练集召回率：{:.2%}'.format(recall_score(y_train, y_train_label)))
print('测试集召回率：{:.2%}'.format(recall_score(y_test, y_test_label)))

print('训练集AUC：{:.2%}'.format(roc_auc_score(y_train, y_train_proba[:,1])))
print('测试集AUC：{:.2%}'.format(roc_auc_score(y_test, y_test_proba[:,1])))
  

### stacking 模型集成

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets.samples_generator import make_blobs


'''创建模型融合中的基模型'''
clfs = [AdaBoostClassifier(),
        RandomForestClassifier(n_estimators=50, n_jobs=-1, criterion='entropy'),
        LogisticRegression (C=0.01),
        ExtraTreesClassifier(n_estimators=50, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

'''对数据集进行切分，切分为训练集和测试集'''

X_train, X_test,y_train,y_test = train_test_split(woe_train_X, train_Y, test_size=0.3)


dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

'''5折stacking'''
n_folds = 5
c, r = y_train.shape
y_train = y_train.values.reshape(c,)
X_train = X_train.values

skf = list(StratifiedKFold(y_train, n_folds))

for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    # print(j, clf)
    dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))
    
    for i, (train, test) in enumerate(skf):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        
        # print("Fold", i)
        X_train_kfold, y_train_kfold, X_test_kfold, y_test_kfold = X_train[train], y_train[train], X_train[test], y_train[test]
        clf.fit(X_train_kfold, y_train_kfold)
        y_submission = clf.predict_proba(X_test_kfold)[:, 1]
        dataset_blend_train[test, j] = y_submission
        
        dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:, 1]
    
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print("使用第" + str(j) + "个模型的：" + "Roc Auc Score: %f" % roc_auc_score(y_test, dataset_blend_test[:, j]))

# stacking 模型融合
clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf.fit(dataset_blend_train, y_train)
y_submission = clf.predict_proba(dataset_blend_test)[:, 1]


print("模型融合的结果：" + "Roc Auc Score: %f" % (roc_auc_score(y_test, y_submission)))
