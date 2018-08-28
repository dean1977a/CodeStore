# 计算正负样本比例  
positive_num = df_train[df_train['label']==1].values.shape[0]  
negative_num = df_train[df_train['label']==0].values.shape[0]  
print(float(positive_num)/float(negative_num))  
