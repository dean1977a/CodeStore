from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from plotly.offline import init_notebook_mode, iplot
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.preprocessing import scale
from lightgbm import LGBMClassifier
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import statsmodels.api as sm
from sklearn.svm import SVC
import plotly.plotly as py
from plotly import tools
from datetime import date
import datetime, time
from pylab import mpl
import seaborn as sns
import pandas as pd
import numpy as np 
import warnings
import random
import pickle
import sys
import gc

%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
folderOfData = 'C:\\Users\\Dean\\Desktop\\model'
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
pd.set_option('max_columns',300) 
pd.set_option('max_row',300)
sys.path.append(folderOfData)
path = "../input/"

def bar_hor(df, col, title, color, w=None, h=None, lm=0, limit=100, return_trace=False, rev=False, xlb = False):
    """
    横向き棒グラフ作成関数
    df:  
    col: 
    title: 
    color: 
    w=None: 
    h=None: 
    lm=0: 
    limit=100: 
    return_trace=False: 
    rev=False: 
    xlb = False:
    """
    cnt_srs = df[col].value_counts()
    yy = cnt_srs.head(limit).index[::-1] 
    xx = cnt_srs.head(limit).values[::-1] 
    if rev:
        yy = cnt_srs.tail(limit).index[::-1] 
        xx = cnt_srs.tail(limit).values[::-1] 
    if xlb:
        trace = go.Bar(y=xlb, x=xx, orientation = 'h', marker=dict(color=color))
    else:
        trace = go.Bar(y=yy, x=xx, orientation = 'h', marker=dict(color=color))
    if return_trace:
        return trace 
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

def bar_hor_noagg(x, y, title, color, w=None, h=None, lm=0, limit=100, rt=False):
    trace = go.Bar(y=x, x=y, orientation = 'h', marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def bar_ver_noagg(x, y, title, color, w=None, h=None, lm=0, rt = False):
    trace = go.Bar(y=y, x=x, marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
def gp(col, title):
    """
    グループ化棒グラフを表示
    col: 表示する列
    title: 図のタイトル
    """
    df1 = app_train[app_train["TARGET"] == 1]
    df0 = app_train[app_train["TARGET"] == 0]
    a1 = df1[col].value_counts()
    b1 = df0[col].value_counts()

    trace1 = go.Bar(x=a1.index, y=a1.values, name='Target : 1', marker=dict(color="#44ff54"))
    trace2 = go.Bar(x=b1.index, y=b1.values, name='Target : 0', marker=dict(color="#ff4444"))

    data = [trace1, trace2]
    layout = go.Layout(barmode='group', height=300, title = title)

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='grouped-bar')
    
def exploreCat(col):
    t = application[col].value_counts()
    labels = t.index
    values = t.values
    colors = ['#96D38C','#FEBFB3']
    trace = go.Pie(labels=labels, values=values,
                   hoverinfo="all", textinfo='value',
                   textfont=dict(size=12),
                   marker=dict(colors=colors,
                               line=dict(color='#fff', width=2)))
    layout = go.Layout(title=col, height=400)
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)

#the realation between the categorical column and the target 
def catAndTrgt(col):
    tr0 = bar_hor(application, col, "Distribution of "+col ,"#f975ae", w=700, lm=100, return_trace= True)
    tr1, tr2 = gp(col, 'Distribution of Target with ' + col)

    fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles = [col +" Distribution" , "% Rpyment difficulty by "+col ,"% of otherCases by "+col])
    fig.append_trace(tr0, 1, 1);
    fig.append_trace(tr1, 1, 2);
    fig.append_trace(tr2, 1, 3);
    fig['layout'].update(height=350, showlegend=False, margin=dict(l=50));
    iplot(fig);
    
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
# Missing values statistics
# Find one unqiue value
def find_one_unqiue_values(df):
# We will takeoff all columns where we have a unique value (constants)
# It is useful because this columns don't give us none information
   discovering_consts = [col for col in df.columns if df[col].nunique() == 1]
   return discovering_consts
# printing the total of columns dropped and the name of columns 
   print("Columns with just one value: ", len(discovering_consts), "columns")
   print("Name of constant columns: \n", discovering_consts)
# 15.1 显示每个特征的唯一值
# seting the function to show 
def knowningData(df, data_type=object, limit=3): #seting the function with df, 
    n = df.select_dtypes(include=data_type) #selecting the desired data type
    for column in n.columns: #initializing the loop
        print("##############################################")
        print("Name of column ", column, ': \n', "Uniques: ", df[column].unique()[:limit], "\n",
              " | ## Total nulls: ", (round(df[column].isnull().sum() / len(df[column]) * 100,2)),
              " | ## Total unique values: ", df.nunique()[column]) #print the data and % of nulls)
        # print("Percentual of top 3 of: ", column)
        # print(round(df[column].value_counts()[:3] / df[column].value_counts().sum() * 100,2))
print("#############################################")
#split categorical, discrete and numerical features  切分类别变量和数值变量
def feature_type_split(data,special_num_list=[],special_cat_list=[]):
    cat_list = []
    dis_num_list = []
    num_list = []
    for i in data.columns.tolist():
        if data[i].dtype == 'object':
            cat_list.append(i)
        elif i in special_cat_list:     # if you want to add some special cases
            cat_list.append(i)
        elif i in special_num_list:     # if you want to add some special cases
            num_list.append(i)
        elif data[i].nunique() < 10:
            dis_num_list.append(i)
        else:
            num_list.append(i)
    return cat_list, dis_num_list, num_list
#示例
categorical_var, dis_num_list, numerical_var = feature_type_split(trainData, special_num_list=['回溯_近7日APP使用次数',],
special_cat_list=['HAS_CHILD','BJ_HIT_RULE','BJ_BLACK_LIST','BAIDU_BLACK_LIST'])



# 数据预处理
# 1. 读取数据：
data_macro = pd.read_csv("macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

# 1.1多条件筛选方法一
f = (footballers
         .loc[footballers['Position'].isin(['ST', 'GK'])]
         .loc[:, ['Value', 'Overall', 'Aggression', 'Position']]
    )
f = f[f["Overall"] >= 80]
f = f[f["Overall"] < 85]
f['Aggression'] = f['Aggression'].astype(float)

# 1.2多条件筛选方法二
data[(data['CREDIT_TYPE']=='Consumer credit')&(data['AMT_ANNUITY']>0)]['AMT_ANNUITY'].sample(100)

# 1.3多条件筛选方法三
#在导入数据阶段直接进行筛选作业
# Define a list of models that we want to review 定义感兴趣的数据关键字
models = ["toyota","nissan","mazda", "honda", "mitsubishi", "subaru", "volkswagen", "volvo"]

# Create a copy of the data with only the top 8 manufacturers 创建子集
df = df_raw[df_raw.make.isin(models)].copy()
#等同于
df = df_raw[df_raw['make'].isin(models)].copy()

# 1.3剔除异常值
def outlier_processing(df,col):
    s=df[col]
    oneQuoter=s.quantile(0.25)
    threeQuote=s.quantile(0.75)
    irq=threeQuote-oneQuoter
    min=oneQuoter-1.5*irq
    max=threeQuote+1.5*irq
    df=df[df[col]<=max]
    df=df[df[col]>=min]
    return df

# 2. 显示为object的属性：
data_train.dtypes[data_train.dtypes=='object']

# 3. 改变数据类型
data_train['material'] = data_train['material'].astype('object')

# 3.1 类型转换
#总结一下astype()函数有效的情形：
#数据列中的每一个单位都能简单的解释为数字(2, 2.12等）
#数据列中的每一个单位都是数值类型且向字符串object类型转换
#如果数据中含有缺失值、特殊字符astype()函数可能失效。
df['Date of Publication'] = pd.to_numeric(extr)

# 3.2
def convert_currency(value):
 """
 转换字符串数字为float类型
 - 移除 ￥ ,
 - 转化为float类型
 """
 new_value = value.replace(',', '').replace('￥', '')
 return np.float(new_value)
#示例
data['2016'].apply(convert_currency)
#等同于
data['2016'].apply(lambda x: x.replace('￥', '').replace(',', '')).astype('float')

# 3.3
def convert_percent(value):
 """
 转换字符串百分数为float类型小数
 - 移除 %
 - 除以100转换为小数
 """
 new_value = value.replace('%', '')
 return float(new_value) / 100
#示例
data['增长率'].apply(convert_percent)
#等同于
data['增长率'].apply(lambda x: x.replace('%', '')).astype('float') / 100

# 3.4直接在读取文件时进行转换
data2 = pd.read_csv("data.csv",
   converters={
    '客户编号': str,
    '2016': convert_currency,
    '2017': convert_currency,
    '增长率': convert_percent,
    '所属组': lambda x: pd.to_numeric(x, errors='coerce'),
    '状态': lambda x: np.where(x == "Y", True, False)
    },
   encoding='gbk')

# 3.5将数值类型特征和类别特征输出出来
def type_features(data):
    categorical_features = data.select_dtypes(include = ["object"]).columns
    numerical_features = data.select_dtypes(exclude = ["object"]).columns
    print( "categorical_features :",categorical_features)
    print('-----'*40)
    print("numerical_features:",numerical_features)

# 3.6将object改变为category
for c in X.columns:
    col_type = X[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        X[c] = X[c].astype('category')



# 4. 概览数据
data_train.describe(include=['object'])
# 5. 合并两个表（上下）
data_all = pd.concat([data_train, data_test], ignore_index=True)

# 6. 合并两个表（左右）
data_all = pd.merge(data_all, data_macro, on='timestamp', how='left')

# 7. 提取Number， Object特征：
object_columns =  data_all.columns[data_all.dtypes == 'object']
number_columns = data_all.columns[data_all.dtypes != 'object']


# 7.1 查看object类的列各列的类别数
df.select_dtypes(['object']).apply(pd.Series.nunique, axis = 0)

# 8. 计算两个特征平均
sa_price = train_df.groupby('sub_area')[['work_share', 'price_doc']].mean()
# 8.1 计算每个子类在其类别中所占的比例
df["Order_Total"] = df.groupby('order')["ext price"].transform('sum') #根据类别分别求和
df["Percent_of_Order"] = df["ext price"] / df["Order_Total"]          #计算子类在类别中的占比

# 8.2 分组计算逾期情况
def ratio(df_temp):
    df_temp['逾期率']=df_temp['M0']/df_temp['立项编号']
    df_temp['拖车率']=df_temp['TARGET']/df_temp['立项编号']
    return df_temp
df_temp= df_2018.groupby(['学历'])['立项编号','M0','TARGET'].count().apply(ratio,axis=1)
# 8.3 绘制逾期分析表
# df：表名    
# var：需要做分类汇总的列名
# sort：boolean是否排序
def groupby_fun(df,var,sort):
    def ratio(df_temp):   
        df_temp['逾期率']=df_temp['M0']/df_temp['立项编号']
        df_temp['3个月内拖车率']=df_temp['3个月内拖车']/df_temp['立项编号']
        #df_temp['6个月内拖车率']=df_temp['6个月内拖车']/df_temp['立项编号']
        df_temp['拖车率']=df_temp['TARGET']/df_temp['立项编号']
        return df_temp
    df_temp_1 = df.groupby(var)['立项编号','M0','3个月内拖车','TARGET'].count().apply(ratio,axis=1)
    df_temp_2 = pd.DataFrame(df_temp_1['立项编号']/ np.sum(df_temp_1['立项编号'])).rename(columns={'立项编号':'占全体交车比率'})
    df_temp_all = pd.merge(df_temp_1,df_temp_2,left_index=True,right_index=True)  
    df_temp_all.rename(columns={'立项编号':'交车数','TARGET':'拖车','M0':'逾期'},inplace=True)
    order = ['交车数', '占全体交车比率', '逾期', '3个月内拖车', '拖车', '逾期率', '3个月内拖车率', '拖车率']
    df_temp_all = df_temp_all[order] 
    if sort == True:
        df_temp_all = df_temp_all.sort_values('逾期率',ascending=False)  
    return df_temp_all
df = groupby_fun(df,'分公司',True)
#行求和
df.loc['Row_sum'] = df.apply(lambda x: x.sum())
#groupby取最大值
get_max = lambda x: x.value_counts(dropna=False).index[0]
# python就是灵活啊。
get_max.__name__ = "most frequent"
df.agg({'ext price': ['sum', 'mean'], 
        'quantity': ['sum', 'mean'], 
        'unit price': ['mean'], 
        'sku': [get_max]})

# 8.4 手工统计信息函数
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

# 8.5 查看类别属性占比
housing["income_cat"].value_counts() / len(housing)

#9 手工匹配字符串并map或者apply示例
def getCabinLetter(cabin):
    match = re.compile("([a-zA-Z]+)").search(cabin)
    if match:
        return  match.group()
    else:
        return 'U'

def getCabinNumber(cabin):
    match = re.compile("([0-9]+)").search(cabin)
    if match:
        return match.group()
    else:
        return 0

#最后有用的属性就只有CabinLetter 和CabinNumber_scaled
def processCabin(df):
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    df['CabinLetter'] = df['Cabin'].map(lambda x:getCabinLetter(x))
    df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]
    df['CabinNumber'] = df['Cabin'].map(lambda x:getCabinNumber(x)).astype(int) + 1 #0太多
    #std比较大所以我们要标准化
    sclar = preprocessing.StandardScaler()
    df['CabinNumber_scaled'] = sclar.fit_transform(df['CabinNumber'])
    return df


# 10. 替换规则
#匹配字符串后进行替换
def replace_titles(x):
    title = x['Title']
    if title in ['Mr', 'Don', 'Major', 'Capt', 'Sir', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Jonkheer']:
        return 'Master'
    elif title in ['Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'male':
            return 'Mr'
        else:
            return 'Mrs'
    elif title == '':
        if x['Sex'] == 'male':
            return 'Master'
        else:
            return 'Miss'
    else:
        return title
df['Title'] = df.apply(replace_titles,axis=1)

# 10.1 根据列中数字大小或字符内容，新建对应的标签列
def majority(x):
    if x > 17:
        return True
    else:
        return False
#示例
stud_alcoh['legal_drinker'] = stud_alcoh['age'].apply(majority)

#为了转换状态列，可以使用Numpy中的where函数，把值为Y的映射成True,其他值全部映射成False,False可不填，这样只有在True的时候才会改变值。
data['状态'] = np.where(data['状态'] == 'Y', True, False)
#示例1：
df['6期以内拖车'] = np.where((df['1128是否拖车']=='是')&(df['已收期数']<6),1,0)
#示例2：
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
#示例3：
df['sex']=np.where(df['name'].isin(['Lucy','Lily']),'girl','boy') #isin的选择方式
#示例4：
df['冰鉴分区间'] = np.where((df['冰鉴分']==-2),'命中规则二',pd.cut(df['冰鉴分'],bj_score_bins))
#示例5：
df1["time"] = np.where(df1["通话时长"].str.contains("分"),df1["通话时长"],["0分"]+df1["通话时长"])


# 11.1 正则表达式
#提取字符串前4位
extr = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)

# 11.2 字符串分割
#示例1
first = dat_edge_1_weight['first'].str.split(':',expand=True,)
#示例2
#df['承租人'].str.split('-',expand=True)这里将Series根据“-”分割成3列，然后再与原来的表格merge
df_final = pd.merge(df,pd.DataFrame(df['承租人'].str.split('-',expand=True)),how='left',left_index=True,right_index=True)

# 12.1 文本处理 
# convert to lower case  转小写
#方法1
suicide_attacks['City'] = suicide_attacks['City'].str.lower()
suicide_attacks['City'] = suicide_attacks['City'].str.upper()
#方法2
loandata[‘term’]=loandata[‘term’].map(str.upper)
loandata[‘term’]=loandata[‘term’].map(str.lower)
loandata[‘term’]=loandata[‘term’].map(str.title)
# remove trailing white spaces   去空格
suicide_attacks['City'] = suicide_attacks['City'].str.strip()
# 替换指定字符
df_final['手机号'] = df_final['手机号'].str.replace('?',' ')
df_final['手机号'] = df_final['手机号'].str.strip()
#将最后一列每个数字前加上Depth
df["new"] =[ 'Depth % i' % i for i in df["Depth"]]

# 12.2 文本清洗，剔除空格，匹配接近的单词
# 包依赖： fuzzywuzzy 
# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    
    # let us know the function's done
    print("All done!")
#应用示例
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="d.i khan")


# 12. 计算两个日期之间的时间差
def dataInterval(data1,data2):
    d1 = datetime.datetime.strptime(data1, '%Y-%m-%d')
    d2 = datetime.datetime.strptime(data2, '%Y-%m-%d')
    delta = d1 - d2
    return delta.days

def getInterval(arrLike):  #用来计算日期间隔天数的调用的函数
    PublishedTime = arrLike['PublishedTime']
    ReceivedTime = arrLike['ReceivedTime']
#    print(PublishedTime.strip(),ReceivedTime.strip())
    days = dataInterval(PublishedTime.strip(),ReceivedTime.strip())  #注意去掉两端空白
    return days

def getInterval_new(arrLike,before,after):  #用来计算日期间隔天数的调用的函数
    before = arrLike[before]
    after = arrLike[after]
#    print(PublishedTime.strip(),ReceivedTime.strip())
    days = dataInterval(after.strip(),before.strip())  #注意去掉两端空白
    return days

if __name__ == '__main__':    
    fileName = "NS_new.xls";
    df = pd.read_excel(fileName) 
    df['TimeInterval'] = df.apply(getInterval , axis = 1)
    df['TimeInterval'] = df.apply(getInterval_new , axis = 1, args = ('ReceivedTime','PublishedTime'))    #调用方式一
    #下面的调用方式等价于上面的调用方式
    df['TimeInterval'] = df.apply(getInterval_new , axis = 1, **{'before':'ReceivedTime','after':'PublishedTime'})  #调用方式二
    #下面的调用方式等价于上面的调用方式
    df['TimeInterval'] = df.apply(getInterval_new , axis = 1, before='ReceivedTime',after='PublishedTime')  #调用方式三
         
#12.1  文本与时间格式转换
#示例1：string变成datetime格式 
dates = pd.to_datetime(pd.Series([‘20010101’, ‘20010331’]), format = ‘%Y%m%d’) 
#示例2：datetime变回string格式 
dates.apply(lambda x: x.strftime(‘%Y-%m-%d’))
#示例3：将timedelta64[ns]转换为float64
df['处置时间'] = df['处置时间'].dt.days


# 13. 用随机森林处理年龄缺失示例
def setMissingData(df,features=[],missFeature='Age'):
    feature_df = df[features]
    X = feature_df[df[missFeature].notnull()].as_matrix()[:,1::]
    y = feature_df[df[missFeature].notnull()].as_matrix()[:,0]
    rtr = RandomForestRegressor(n_estimators=2000,n_jobs=-1)#无限制处理机
    rtr.fit(X,y)
    predicitedAges = rtr.predict(feature_df[df[missFeature].isnull()].as_matrix()[:,1:])
    df.loc[(df[missFeature].isnull()),missFeature] = predicitedAges
    return df

# def setMissingAges(df):
#
#     age_df = df[['Age', 'Embarked', 'Fare', 'Parch', 'SibSp', 'Title_id', 'Pclass', 'Names', 'CabinLetter']]
#     X = age_df.loc[(df.Age.notnull())].values[:, 1::]
#     y = age_df.loc[(df.Age.notnull())].values[:, 0]
#
#     rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
#     rtr.fit(X, y)
#
#     predictedAges = rtr.predict(age_df.loc[(df.Age.isnull())].values[:, 1::])
#     df.loc[(df.Age.isnull()), 'Age'] = predictedAges
#     return  df
def processAge(df):
    #先填缺省值
    #预测的方法RandomForest
    df = setMissingData(df, features=['Age','Embarked','Fare', 'Parch', 'SibSp', 'Title_id','Pclass','Names','CabinLetter'], missFeature='Age')
    #df = setMissingAges(df)
    #此处用中位数以及均值填充但是需要先分层再求均值。
    # mean_master = np.average(df['Age'][df.Title=='Master'].dropna())
    # mean_mr = np.average(df['Age'][df.Title=='Mr'].dropna())
    # mean_miss = np.average(df['Age'][df.Title=='Miss'].dropna())
    # mean_mrs = np.average(df['Age'][df.Title=='Mrs'].dropna())
    # df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age'] = mean_master
    # df.loc[(df.Age.isnull()) & (df.Title == 'Mr'), 'Age'] = mean_mr
    # df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age'] = mean_miss
    # df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'), 'Age'] = mean_mrs
    scaler = preprocessing.StandardScaler()
    df['Age_scaled'] = scaler.fit_transform(df['Age'])
    #特别提到老人小孩。那么显然要离散化年龄
    # bin into quartiles and create binary features
    #按照频率接近的类别编号在一起
    df['Age_bin'] = pd.qcut(df['Age'],4)
    #而若只跟几个年龄段有关跟其他无关那么虚拟化要
    df = pd.concat([df, pd.get_dummies(df['Age_bin']).rename(columns=lambda x: 'Age_' + str(x))], axis=1)

    df['Age_bin_id'] = pd.factorize(df['Age_bin'])[0]+1
    #Age_bin_id也要标准化为了后续组合以及PCA方便
    scaler = preprocessing.StandardScaler()
    df['Age_bin_id_scaled'] = scaler.fit_transform(df['Age_bin_id'])
    df['Child'] = (df['Age']<13).astype(int)

    #变化不大
    # from sklearn import  preprocessing
    # scaler = preprocessing.StandardScaler()
    # df['Age_bin_id_scaled'] = scaler.fit_transform(df['Age_bin_id'])
    return  df

#14 利用数据透视表生成区间特征
def get_bureau_balance():   
    data = pd.read_csv(path +'bureau_balance.csv')
    data["STATUS"], uniques = pd.factorize(data["STATUS"]) 
    data["MONTHS_BALANCE"] = abs(data["MONTHS_BALANCE"])
    cut_points = [0,2,4,12,24,36]                                 #选择区间切分点
    cut_points = cut_points + [data["MONTHS_BALANCE"].max()]
    labels = ["2MON","4MON","12MON","24MON","36MON","ABOVE"]      #这里可以改成1,3,6,12,18,24,36,above
    #构建区间标签
    data["MON_INTERVAL"] = pd.cut(data["MONTHS_BALANCE"], cut_points,labels=labels,include_lowest=True)     
    feature = pd.pivot_table(data,index=["SK_ID_BUREAU"],columns=["MON_INTERVAL"],values=["STATUS"],aggfunc=[np.max,np.mean,np.std]).astype('float32')
    #将3级标签转换为1级标签
    feature.columns = ["_".join(f_).upper() for f_ in feature.columns]
    
    bb_agg = data.groupby('SK_ID_BUREAU').agg({'MONTHS_BALANCE': ['min', 'max', 'size']}).astype('float32')
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    feature = pd.merge(feature,bb_agg,how="left",left_index=True,right_index=True)
    return feature, feature.columns.tolist()

# 15 查找只含有一个唯一值的特征
def find_one_unqiue_values(df):
# We will takeoff all columns where we have a unique value (constants)
# It is useful because this columns don't give us none information
   discovering_consts = [col for col in df.columns if df[col].nunique() == 1]
   return discovering_consts
# printing the total of columns dropped and the name of columns 
   print("Columns with just one value: ", len(discovering_consts), "columns")
   print("Name of constant columns: \n", discovering_consts)
# 15.1 显示每个特征的唯一值
# seting the function to show 
def knowningData(df, data_type=object, limit=3): #seting the function with df, 
    n = df.select_dtypes(include=data_type) #selecting the desired data type
    for column in n.columns: #initializing the loop
        print("##############################################")
        print("Name of column ", column, ': \n', "Uniques: ", df[column].unique()[:limit], "\n",
              " | ## Total nulls: ", (round(df[column].isnull().sum() / len(df[column]) * 100,2)),
              " | ## Total unique values: ", df.nunique()[column]) #print the data and % of nulls)
        # print("Percentual of top 3 of: ", column)
        # print(round(df[column].value_counts()[:3] / df[column].value_counts().sum() * 100,2))
print("#############################################")

#16 将groupby之后的两层列名合并成一层
temp.columns = ["_".join(x) for x in temp.columns.ravel()]



# 数据可视化
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

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

# 9. 画出每个特征的散点图和柱状图
#示例1
from scipy import stats
def plot_dist(df, feature, pic_name='dist_plot.png'):
    fcols = 2
    frows = len(feature) + 1
    print(fcols, frows)
    plt.figure(figsize=(5*fcols, 4*frows))

    i = 0
    for col in feature:
        
        i += 1
        ax = plt.subplot(frows, fcols, i)

        plt.scatter(df[col], df['TARGET'])

        plt.xlabel(col)
        plt.ylabel('TARGET')

        i += 1
        ax = plt.subplot(frows, fcols, i)
        sns.distplot(df[col].dropna(), fit=stats.norm)
        plt.xlabel(col)

    plt.tight_layout()
#运行示例    
plot_dist(data, data.columns)

示例2
g = sns.pairplot(df, hue="target", palette="husl")

#10.1 绘制单个特征的分布图
def plotHist(df,nameOfFeature):
    cls_train = df[nameOfFeature]
    data_array = cls_train
    hist_data = np.histogram(data_array)
    binsize = .5

    trace1 = go.Histogram(
        x=data_array,
        histnorm='count',
        name='Histogram of Wind Speed',
        autobinx=False,
        xbins=dict(
            start=df[nameOfFeature].min()-1,
            end=df[nameOfFeature].max()+1,
            size=binsize
        )
    )

    trace_data = [trace1]
    layout = go.Layout(
        bargroupgap=0.3,
         title='The distribution of ' + nameOfFeature,
        xaxis=dict(
            title=nameOfFeature,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Number of labels',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
    fig = go.Figure(data=trace_data, layout=layout)
    py.iplot(fig)

#10.1 绘制两个特征对比分布图
from scipy.stats import skew
from scipy.stats import kurtosis
def plotBarCat(df,feature,target):
    
    
    
    x0 = df[df[target]==0][feature]
    x1 = df[df[target]==1][feature]

    trace1 = go.Histogram(
        x=x0,
        opacity=0.75
    )
    trace2 = go.Histogram(
        x=x1,
        opacity=0.75
    )

    data = [trace1, trace2]
    layout = go.Layout(barmode='overlay',
                      title=feature,
                       yaxis=dict(title='Count'
        ))
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='overlaid histogram')
    
    def DescribeFloatSkewKurt(df,target):
        """
            A fundamental task in many statistical analyses is to characterize
            the location and variability of a data set. A further
            characterization of the data includes skewness and kurtosis.
            Skewness is a measure of symmetry, or more precisely, the lack
            of symmetry. A distribution, or data set, is symmetric if it
            looks the same to the left and right of the center point.
            Kurtosis is a measure of whether the data are heavy-tailed
            or light-tailed relative to a normal distribution. That is,
            data sets with high kurtosis tend to have heavy tails, or
            outliers. Data sets with low kurtosis tend to have light
            tails, or lack of outliers. A uniform distribution would
            be the extreme case
        """
        print('-*-'*25)
        print("{0} mean : ".format(target), np.mean(df[target]))
        print("{0} var  : ".format(target), np.var(df[target]))
        print("{0} skew : ".format(target), skew(df[target]))
        print("{0} kurt : ".format(target), kurtosis(df[target]))
        print('-*-'*25)
    
    DescribeFloatSkewKurt(df,target)
#运行示例
plotBarCat(df,df_name[0],'Outcome')





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

# 6.1 分组来填补平均值
for num in number_columns:
    if(sum(data_all[num].isnull())>0):
        isnull_raw = data_all[num].isnull()
        isnull_yearmonth = data_all.ix[isnull_raw, 'yearmonth'].values
        data_all_groupby[num].transform(lambda x: x.fillna(x.mean()))

# 6.2 利用map进行填充分组的平均值
temp = data.groupby("CREDIT_TYPE")["AMT_ANNUITY"].mean()
data["CREDIT_TYPE_AMT_ANNUITY"] = data["CREDIT_TYPE"].map(temp)

# 7. Get_dummies离散化
dummies = pd.get_dummies(data=data_all[ob], prefix="{}#".format(ob))
data_all.drop(ob, axis=1, inplace=True)
data_all = data_all.join(dummies)


# 7.1 
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns



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

# 11. 绘制学习曲线，以确定模型的状况是否过拟合和欠拟合
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,
                        n_jobs=None,train_sizes=np.linspace(.1,1.0,10)):
    """
    生成训练和测试的学习曲线图

    参数:
    ---------------------
    estimator: object type

    title: string
           图表的标题

    X: 类数组，形状(n_samples, n_features)

       训练向量，其中n_samples为样本个数,n_features是特性的数量。


    y: 类数组，形状(n_samples)或(n_samples, n_features)，可选目标相对于X进行分类或回归;


    ylim:元组，形状(ymin, ymax)，可选定义绘制的最小和最大y值。


    cv:int，交叉验证生成器或可迭代的，可选的确定交叉验证拆分策略。

        cv的可能输入是:

            -无，使用默认的3倍交叉验证，

            -整数，指定折叠的次数。

    n_jobs:int或None，可选(默认=None) 并行运行的作业数。'None'的意思是1。
           “-1”表示使用所有处理器。


    train_sizes：类数组，形状(n_ticks，)， dtype float或int

                相对或绝对数量的训练例子，将用于生成学习曲线。如果dtype是float，则将其视为

                训练集的最大大小的分数(这是确定的)，即它必须在(0,1)范围内。
                否则，它被解释为训练集的绝对大小。

                注意，为了分类，样本的数量通常必须足够大，可以包含每个类的至少一个示例。
                (默认:np.linspace(0.1, 1.0, 5))
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
import matplotlib.pyplot as plt

from sklearn.datasets import  make_gaussian_quantiles
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
# ##########################
# 生成2维正态分布，生成的数据按分位数分为两类，50个样本特征，5000个样本数据
X,y = make_gaussian_quantiles(cov=2.0,n_samples=5000,n_features=50,
                              n_classes=2,random_state=1)
# 设置一百折交叉验证参数，数据集分层越多，交叉最优模型越接近原模型
cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=1)
# 分别画出CART分类决策树和AdaBoost分类决策树的学习曲线
estimatorCart = DecisionTreeClassifier(max_depth=1)
estimatorBoost = AdaBoostClassifier(base_estimator=estimatorCart,
                                    n_estimators=270)
# 画CART决策树和AdaBoost的学习曲线
estimatorTuple = (estimatorCart,estimatorBoost)
titleTuple =("decision learning curve","adaBoost learning curve")
title = "decision learning curve"
for i in range(2):
    estimator = estimatorTuple[i]
    title = titleTuple[i]
    plot_learning_curve(estimator,title, X, y, cv=cv)
    plt.show()
 


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

#############################################################################################
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


#模型评价
#KS值计算
#ks_calc_cross可以无视数据中是否有NAN值，ks_calc_auc则不行
#crosstab实现
def ks_calc_cross(data,score_col,class_col):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    score_col: 一维数组或series，代表模型得分（一般为预测正类的概率）
    class_col: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值，'crossdens': 好坏人累积概率分布以及其差值gap
    '''
    ks_dict = {}
    crossfreq = pd.crosstab(data[score_col[0]],data[class_col[0]])
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['gap'] = abs(crossdens[0] - crossdens[1])
    ks = crossdens[crossdens['gap'] == crossdens['gap'].max()]
    return ks,crossdens


#roc_curve实现
from sklearn.metrics import roc_curve,auc
def ks_calc_auc(data,score_col,class_col):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    score_col: 一维数组或series，代表模型得分（一般为预测正类的概率）
    class_col: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值
    '''
    fpr,tpr,threshold = roc_curve((1-data[class_col[0]]).ravel(),data[score_col[0]].ravel())
    ks = max(tpr-fpr)
    return ks
###############################################################################################
#绘制多个模型对比
from sklearn.metrics 
import accuracy_score, precision_score, recall_score, f1_score from sklearn.metrics 
import roc_auc_score, roc_curve, auc 
import matplotlib.pyplot as plt 
def model_metrics(clf, X_train, X_test, y_train, y_test): 
    # 预测 
    y_train_pred = clf.predict(X_train) 
    y_test_pred = clf.predict(X_test) 
    y_train_proba = clf.predict_proba(X_train)[:, 1] 
    y_test_proba = clf.predict_proba(X_test)[:, 1] 
    # 准确率 
    print('[准确率]', end=' ') 
    print('训练集：', '%.4f' % accuracy_score(y_train, y_train_pred), end=' ') 
    print('测试集：', '%.4f' % accuracy_score(y_test, y_test_pred)) 
    # 精准率 
    print('[精准率]', end=' ') 
    print('训练集：', '%.4f' % precision_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % precision_score(y_test, y_test_pred)) 
    # 召回率 
    print('[召回率]', end=' ') 
    print('训练集：', '%.4f' % recall_score(y_train, y_train_pred), end=' ') 
    print('测试集：', '%.4f' % recall_score(y_test, y_test_pred)) 
    # f1-score 
    print('[f1-score]', end=' ') 
    print('训练集：', '%.4f' % f1_score(y_train, y_train_pred), end=' ') 
    print('测试集：', '%.4f' % f1_score(y_test, y_test_pred)) 
    # auc取值：用roc_auc_score或auc 
    print('[auc值]', end=' ') 
    print('训练集：', '%.4f' % roc_auc_score(y_train, y_train_proba), end=' ') 
    print('测试集：', '%.4f' % roc_auc_score(y_test, y_test_proba)) 
    # roc曲线 
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_proba, pos_label=1) 
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_proba, pos_label=1) 
    label = ["Train - AUC:{:.4f}".format(auc(fpr_train, tpr_train)), "Test - AUC:{:.4f}".format(auc(fpr_test, tpr_test))] 
    plt.plot(fpr_train, tpr_train) 
    plt.plot(fpr_test, tpr_test) 
    plt.plot([0, 1], [0, 1], 'd--') 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.legend(label, loc=4) 
    plt.title("ROC curve") 
    plt.show() 
if __name__ == "__main__": 
    model_metrics(log_reg, X_train, X_test, y_train, y_test) 
    model_metrics(dtree, X_train, X_test, y_train, y_test) 
    model_metrics(LinearSVC, X_train, X_test, y_train, y_test) 
    model_metrics(xgbClassifier, X_train, X_test, y_train, y_test) 
    model_metrics(lgbmClassifier, X_train, X_test, y_train, y_test)

############################################################################################################################
#贝叶斯参数优化
from bayes_opt import BayesianOptimization

X = application_train.drop('TARGET', axis=1)
y = application_train.TARGET
def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, categorical_feature = categorical_feats, free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    return lgbBO.res['max']['max_params']

