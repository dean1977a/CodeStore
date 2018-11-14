import math
import pandas as pd
    """
        inputfile:dataframe所在输入文件
        feture:需要分析的特征变量
        sep 分段表达式
        target y变量
        """
##example iv,df=iv_value('E:/breast_cancer.csv','radius_mean',(10,15,20),'diagnosis')
def iv_value(file,feature,sep,target):
    ###sep格式为（10，15，20）
    data = pd.read_csv(file, sep=',')
    #data = pd.read_csv('E:/breast_cancer.csv', sep=',')
    # woe
    #sep_value = sep.split(',')
    sep_value =str(sep).replace('(','').replace(')','').split(',')
    sep_len = len(sep_value)
    dict_bin = {}
    class_bin = {}
    len_dict_bin = {}
    len_dict_bin_0 = {}
    len_dict_bin_1 = {}
    woe_bin = {}
    iv_bin = {}
    if sep_len == 1:
        dict_bin[0] = data.loc[data[feature] <= float(sep_value[0]), :]
        dict_bin[1] = data.loc[data[feature] > float(sep_value[0]), :]
        dict_bin[2] = sum(data[feature].isnull())
        len_dict_bin[0] = len(dict_bin[0])
        len_dict_bin[1] = len(dict_bin[1])
        len_dict_bin[2] = len(dict_bin[2])
        class_bin[0] = "(0," + sep_value[0] + "]"
        class_bin[1] = "(" + sep_value[0] + "...)"
        class_bin[2] = "NA"
    else:
        for index, item in enumerate(sep_value):####区间
            if index == 0:
                dict_bin[0] = data.loc[data[feature] <= float(item), :]
                len_dict_bin[0] = len(dict_bin[0])
                class_bin[0] = "(0," + str(float(item)) + "]"
            else:
                dict_bin[index] = (
                    data.loc[(data[feature] >= float(sep_value[index - 1])) & (data[feature] < float(item)),
                    :])
                len_dict_bin[index] = len(dict_bin[index])
                class_bin[index] = "(" + str(sep_value[index - 1]) + "," + str(sep_value[index]) + "]"
        dict_bin[index + 1] = data.loc[data[feature] > float(item), :]
        dict_bin[index + 2] = data.loc[data[feature].isnull()]
        len_dict_bin[index + 1] = len(dict_bin[index + 1])
        len_dict_bin[index + 2] = len(dict_bin[index + 2])
        class_bin[index + 1] = "(" + str(sep_value[index]) + "...)"
        class_bin[index + 2] = "NA"

    for index, item in enumerate(dict_bin):
        len_dict_bin_0[index] = len(dict_bin[index][dict_bin[index][target] == 0])
        len_dict_bin_1[index] = len(dict_bin[index][dict_bin[index][target] == 1])

    len_data_0 = len(data[data[target] == 0])
    len_data_1 = len(data[data[target] == 1])
    for index, item in enumerate(dict_bin):
        try:
            woe_bin[index] = math.log(math.e, (float(len_dict_bin_1[index]) / float(len_data_1)) / (
                float(len_dict_bin_0[index]) / float(len_data_0)))
            iv_bin[index] = ((float(len_dict_bin_1[index]) / float(len_data_1)) - (
                float(len_dict_bin_0[index]) / float(len_data_0))) * math.log(math.e, (
                float(len_dict_bin_1[index]) / float(len_data_1)) / (float(len_dict_bin_0[index]) / float(len_data_0)))
        except Exception as e:
            iv_bin[index] = 0
    iv_sum=0.0
    for key in iv_bin:
        try:
            iv_sum=iv_sum+float(iv_bin[key])
        except Exception as e:
            print (e)
    return iv_sum
            
        
    dict_result = {}
    len_dict_bin_0[" "] = len_data_0
    len_dict_bin_1[" "] = len_data_1
    woe_bin[" "] = ""
    iv_bin[" "]=sum(iv_bin.values())
    class_bin[" "] = ""
    len_dict_bin[" "] = len(data)
    dict_result["bad"] = len_dict_bin_0
    dict_result["good"] = len_dict_bin_1
    dict_result["all"] = len_dict_bin
    dict_result["woe"] = woe_bin
    dict_result["iv"] = iv_bin
    dict_result["class"] = class_bin
    df = pd.DataFrame(dict_result)

    dict_result["%good"] = (df['good'] / df['all']).map('{:.2%}'.format);
    dict_result["%bad"] = (df['bad'] / df['all']).map('{:.2%}'.format);
    df["%good"] = dict_result["%good"]
    df["%bad"] = dict_result["%bad"]

    # 调整列的顺序
    df = df.ix[:, ['class', 'good', 'bad', '%good', '%bad', 'all', 'woe', 'iv']]
    #print df
    return iv_sum,df
