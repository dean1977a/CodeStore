-----------------------------缺失值统计---------------------------------------------
# This function use to print feature with null values and null count
#计算缺失值方法一
def checkNullValue(df):
    nullColumnsStat = []
    numRows = df.count()
    for c in tqdm(df.columns):
        nullRows = df.filter(col(c).isNull()).count()
        #print(c,nullRows)
        if( nullRows > 0):
            nullRatio = round((nullRows/numRows),3)
            temp = c,nullRatio
            nullColumnsStat.append(temp)
    return nullColumnsStat

null_columns_count_list = checkNullValue(train)
pd.DataFrame(null_columns_count_list,columns=['Column_With_Null_Value', 'Null_Values_Ratio']) \
              .sort_values(by='Null_Values_Ratio',ascending=False)

#方法二
train_miss = train.agg(*[(1-(count(c)/count('*'))).alias(c + '_missing') for c in train.columns])\
             .toPandas().T.sort_values(by=0,ascending=False)
train_miss.head()

-------------------------------------列分类-------------------------------------------
#计算量比较大
#根据key进行降采样
def downSampleBy(df, key):
    dataSize = df.count()
    while dataSize >1000000:
        negativeRatio = 0.1
        if   dataSize > 10000000:
            positiveRatio = 0.01
        else:
            positiveRatio = 0.1
        df = df.sampleBy(key, fractions={0: positiveRatio, 1: negativeRatio}, seed=42)
        print("after down sampling the data size is ", df.count())
    print("there is no need to downsampling")
    return df

#split categorical, discrete and numerical features  切分类别变量和数值变量
#通过手工添加special_num_list,special_cat_list排除异常变量
def featureTypeSplit(df, key, specialNumList=[], specialCatList=[],noNeedToSplitList=[]):
    df = downSampleBy(df, key)
    catVarList = []
    numVarList = []
    for colName,colType in dict(df.dtypes).items():
        if colName in noNeedToSplitList:
            continue
        elif colName in specialCatList:     # if you want to add some special cases
            print("catVarList add ", colName)
            catVarList.append(colName)
        elif colName in specialNumList:     # if you want to add some special cases
            print("numVarList add ", colName)
            numVarList.append(colName)
        elif colType == 'string':
            print("catVarList add ", colName)
            catVarList.append(colName)
        elif colType == 'boolean':
            print("catVarList add ", colName)
            catVarList.append(colName)
        elif colType == 'binary':
            print("catVarList add ", colName)
            catVarList.append(colName)
        elif df.agg(F.countDistinct(df.colName)).collect()[0][0] < 1500 :
            print("catVarList add ", colName)
            catVarList.append(colName)
        else:
            print("numVarList add ", colName)
            numVarList.append(colName)
    return catVarList,numVarList


categoricalVar,disNumVar,numericalVar = featureTypeSplit(train,key='isFraud',specialNumList=[],specialCatList=[])

#split categorical, discrete and numerical features切分类别变量和数值变量
#通过手工添加special_num_list,special_cat_list指定特别变量
def featureTypeSplit(df, specialNumList=[], specialCatList=[],noNeedToSplitList=[]):
    catVarList = []
    numVarList = []
    for colName,colType in dict(df.dtypes).items():
        if colName in noNeedToSplitList:
            continue
        elif colName in specialCatList:     # if you want to add some special cases
            print("catVarList add ", colName)
            catVarList.append(colName)
        elif colName in specialNumList:     # if you want to add some special cases
            print("numVarList add ", colName)
            numVarList.append(colName)
        elif colType == 'string':
            print("catVarList add ", colName)
            catVarList.append(colName)
        elif colType == 'boolean':
            print("catVarList add ", colName)
            catVarList.append(colName)
        elif colType == 'binary':
            print("catVarList add ", colName)
            catVarList.append(colName)
        else:
            print("numVarList add ", colName)
            numVarList.append(colName)
    return catVarList,numVarList

#简易版
categoricalVar = list(map(lambda x:x[0]  ,filter( lambda x: x[1] == 'string',train.dtypes)))
numericalVar   = list(map(lambda x:x[0]  ,filter( lambda x: x[1] != 'string',train.dtypes)))


-------------------------------异常值处理----------------------------------------------------

def outlierProcessing(df,col):
    quantiles = df.approxQuantile(col, [0.25, 0.75], 0.05)
    SizeBeforProcess = df.col.count()
    # 计算4分位距
    irq = quantiles[1] - quantiles[0]
    min=quantiles[0]-1.5*irq
    max=quantiles[1]+1.5*irq
    dfAfterProcess = df.filter(df[col]<=max).filter(df[col]>=min)
    SizeAfterProcess = dfAfterProcess.col.count()
    OutlierSize = SizeBeforProcess - SizeAfterProcess
    print('Non-outlier observations: %d' % OutlierSize ) # printing total number of non outlier values
    print("Total percentual of Outliers: ",round((OutlierSize/SizeBeforProcess * 100), 4)) # Percentual of outliers in points
    return dfAfterProcess


--------------------------------------OneHot编码---------------------------------------------------------------------

from pyspark.sql.functions import col, countDistinct, approxCountDistinct
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator

def ohcOneColumn(df, colName, debug=False):

  colsToFillNa = []

  if debug: print("Entering method ohcOneColumn")
  countUnique = df.groupBy(colName).count().count()
  if debug: print(countUnique)

  collectOnce = df.select(colName).distinct().collect()
  for uniqueValIndex in range(countUnique):
    uniqueVal = collectOnce[uniqueValIndex][0]
    if debug: print(uniqueVal)
    newColName = str(colName) + '_' + str(uniqueVal) + '_TF'
    df = df.withColumn(newColName, df[colName]==uniqueVal)
    colsToFillNa.append(newColName)
  df = df.drop(colName)
  df = df.na.fill(False, subset=colsToFillNa)
  return df

def detectAndLabelCat(sparkDf, minValCount=5, debug=False, excludeCols=['Target']):
  if debug: print("Entering method detectAndLabelCat")
  newDf = sparkDf
  colList = sparkDf.columns

  for colName in sparkDf.columns:
    uniqueVals = sparkDf.groupBy(colName).count()
    if debug: print(uniqueVals)
    countUnique = uniqueVals.count()
    dtype = str(sparkDf.schema[colName].dataType)
    #dtype = str(df.schema[nc].dataType)
    if (colName in excludeCols):
      if debug: print(str(colName) + ' is in the excluded columns list.')

    elif countUnique == 1:
      newDf = newDf.drop(colName)
      if debug:
        print('dropping column ' + str(colName) + ' because it only contains one unique value.')
      #end if debug
    #elif (1==2):
    elif ((countUnique < minValCount) | (dtype=="String") | (dtype=="StringType")):
      if debug: 
        print(len(newDf.columns))
        oldColumns = newDf.columns
      newDf = ohcOneColumn(newDf, colName, debug=debug)
      if debug: 
        print(len(newDf.columns))
        newColumns = set(newDf.columns) - set(oldColumns)
        print('Adding:')
        print(newColumns)
        for newColumn in newColumns:
          if newColumn in newDf.columns:
            try:
              newUniqueValCount = newDf.groupBy(newColumn).count().count()
              print("There are " + str(newUniqueValCount) + " unique values in " + str(newColumn))
            except:
              print('Uncaught error discussing ' + str(newColumn))
          #else:
          #  newColumns.remove(newColumn)

        print('Dropping:')
        print(set(oldColumns) - set(newDf.columns))

    else:
      if debug: print('Nothing done for column ' + str(colName))

      #end if countUnique == 1, elif countUnique other condition
    #end outer for
  return newDf


from pyspark.mllib.stat  import Statistics 
def corrFilter(df,col,excludeCols,target):
    useFulCol = []
    corrScore = []
    for col in train.select(col).columns :
        if col not in excludeCols:
            if Statistics.chiSqTest(train.select('C2').collect()).pValue < 0.05:
                colCorr = float(str(train.stat.corr(col,target))[0:5])
                if colCorr > 0.03 or colCorr < -0.03:
                    useFulCol.append(col)
                    corrScore.append(colCorr)
    pearsonTable = pd.DataFrame({'colNmae':useFulCol,'pearson ':corrScore})
    pearsonTable.sort_values(by='spearman',ascending=False, inplace=True)
    return pearsonTable


def plotKde(df,col,target):
    
    # Calculate the correlation coefficient between the new variable and the target
    corr = str(train.stat.corr(col,target))[0:5]
    
    amtValTrue = df.filter(df[target] == 1).select(col,target).na.fill(0.0).toPandas()[col].values
    amtValFalse = df.filter(df[target] == 0).select(col,target).na.fill(0.0).toPandas()[col].values
    
    # Calculate medians for repaid vs not repaid
    medResponse = np.median(amtValTrue)
    medNotResponse = np.median(amtValFalse)
    
    plt.figure(figsize = (12, 6))
    
    # Plot the distribution for target == 0 and target == 1
    
    sns.kdeplot(amtValTrue , label = 'TARGET == 0')
    sns.kdeplot(amtValFalse , label = 'TARGET == 1')
    
    # label the plot
    plt.xlabel(col); plt.ylabel('Density'); plt.title('%s Distribution' % col)
    plt.legend();
    
    # print out the correlation
    print('The correlation between %s and the TARGET is %s' % (col, corr))
    # Print out average values
    print('Median value for loan that was not response = %s' % medNotResponse)
    print('Median value for loan that was response =     %s' % medResponse)

def plotDist(df,col,target,needLog = False):
    if needLog == True:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
        
    amtValTrue = df.filter(df[target] == 1).select(col,target).na.fill(0.0).toPandas()[col].values

    sns.distplot(amtValTrue, ax=ax1, color='r')
    ax1.set_title('Distribution of ' + col + ', target =1', fontsize=14)

    amtValFalse = df.filter(df[target] == 0).select(col,target).na.fill(0.0).toPandas()[col].values

    sns.distplot(amtValFalse, ax=ax2, color='b')
    ax2.set_title('Distribution of ' + col + ', isFraud=0', fontsize=14)
    
    if needLog == True:
        sns.distplot(np.log(amtValTrue), ax=ax3, color='r')
        ax3.set_title('Distribution of LOG ' + col + ', isFraud=1', fontsize=14)

        sns.distplot(np.log(amtValFalse), ax=ax4, color='b')
        ax4.set_title('Distribution of LOG ' + col + ', isFraud=0', fontsize=14)

    plt.show()

def plotProportion(df,filterCondition,y_axis,target):
    fig, (ax1,ax2 )= plt.subplots(2,1,figsize=(10, 20))
    if filterCondition == None:
        tmp=df.select(y_axis,target).na.fill({y_axis:'-9'}).toPandas()
    else:
        tmp=df.filter(filterCondition).select(y_axis,target).na.fill({y_axis:'-9'}).toPandas()
    
    sns.countplot(y=y_axis, data=tmp,ax=ax1)\
                 .set_title(y_axis, fontsize=16)
    
    props = tmp.groupby(y_axis)[target].value_counts(normalize=True).unstack()

    sns.set_palette(['limegreen', 'green'])
    
    props.plot(kind='bar', stacked='True', ax=ax2).set_ylabel('Proportion')
    
    plt.show()

    
def plotViolinBox(df,filterCondition,x_axis,y_axis,target):
    fig, (ax1,ax2 )= plt.subplots(2,1,figsize=(15, 28))
    
    if filterCondition == None:
        tmp=df.select(x_axis,y_axis,target).na.fill({y_axis:'-9'}).toPandas()
    else:
        tmp=df.filter(filterCondition).select(x_axis,y_axis,target).na.fill({y_axis:'-9'}).toPandas()

    sns.violinplot(x=x_axis, y=y_axis, hue=target,
                      data=tmp, palette=['lightgreen', 'green'],
                      split=True, ax=ax1).set_title( x_axis + '  vs. ' + y_axis, fontsize=14)

    sns.boxplot(x=x_axis, y=y_axis, hue=target,
                   data=tmp, palette=['lightgreen', 'green'],
                   ax=ax2).set_title( x_axis + '  vs. ' + y_axis, fontsize=14)

    plt.show()


# 通过ks值检测训练集和测试集差异
# https://blog.csdn.net/simingjiao/article/details/88693650
# https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.stats.ks_2samp.html
def getDiffColumns(trainDF, testDF, cols, showPlots=True, threshold=0.1):
    """ 检查训练集和测试集数据的数据稳定性，通过统计每列的KS值，返回低于阈值的列和高于阈值的列 。
    第一个值是检验统计量，第二个值是p值。如果p值小于0.05（对于5％的显着性水平），则意味着不能拒绝Null-Hypothese 样本分布是相同的。
    :param trainDF: train data set
    :param testDF: test data set
    :param cols: columns need to compute
    :param showPlots: boolean
    :param threshold：0~1 float type , to determine whether the distribution is abnormal
    :return DataFrame of different distribution columns & DataFrame of same distribution columns
    """

    from scipy.stats import ks_2samp
    diffData = []
    conData = []

    for col in cols:

        trainColValues = trainDF.select(col).toPandas()[col].values
        testColValues = testDF.select(col).toPandas()[col].values

        statistic, pvalue = ks_2samp(trainColValues,
                                     testColValues)

        if pvalue <= 0.05 and np.abs(statistic) > threshold:
            diffData.append(
                {'feature': col, 'pvalue': np.round(pvalue, 5), 'statistic': np.round(np.abs(statistic), 2)})
            if showPlots:
                plt.figure(figsize=(8, 4))
                plt.title("Kolmogorov-Smirnov test for train/test\n"
                          "feature: {}, statistics: {}, pvalue: {}".format(col, statistic, pvalue))

                sns.kdeplot(trainColValues, color='blue', shade=True, label='Train')
                sns.kdeplot(testColValues, color='green', shade=True, label='Test')

                plt.tight_layout()
                plt.show()

        else:
            conData.append({'feature': col, 'pvalue': np.round(pvalue, 5), 'statistic': np.round(np.abs(statistic), 2)})

    diffDF = pd.DataFrame(diffData)
    conDF = pd.DataFrame(conData)

    return diffDF, conDF

diffDF,conDF = getDiffColumns(trainDF = train ,testDF = test ,cols = numericalVar ,showPlots=True,threshold=0.1)


