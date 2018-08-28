# 计算正负样本比例  
positive_num = df_train[df_train['label']==1].values.shape[0]  
negative_num = df_train[df_train['label']==0].values.shape[0]  
print(float(positive_num)/float(negative_num))  

#下采样
def down_sample(df):
    """
    Downsampling
    """
    test_df = df[df['TARGET'].isnull()]
    train_df1 = df[df['TARGET'] == 1]
    train_df2 = df[df['TARGET'] == 0]
    train_df3 = train_df2.sample(frac=0.1, random_state=40, replace=True)
    df = pd.concat([test_df, train_df1, train_df3])
    print("df1 shape:", train_df1.shape)
    print("df3 shape:", train_df3.shape)
    print(df[df['TARGET'] == 1].shape)
    del train_df1
    del train_df2
    del train_df3
    del test_df
    gc.collect()

    return df
