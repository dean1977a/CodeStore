#对该列进行强制的字符类型转换
df["token"] = df["token"].astype(str)
#筛选df这个数据集下，token这个字段下面的value字符串长度大于20的
df= df[df['token'].str.len() >20]
