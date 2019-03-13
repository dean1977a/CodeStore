#split categorical, discrete and numerical features  切分类别变量和数值变量
#通过手工添加special_num_list,special_cat_list不断排除异常变量
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
categorical_var, dis_num_list, numerical_var = feature_type_split(trainData, special_num_list=[],special_cat_list=[]) 
