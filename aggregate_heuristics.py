def find_cats_non_cats(df_, grouper):
    
    df = df_.copy().drop(grouper, axis = 1)
    cats = [col for col in df.columns if df[col].dtype == "object"]
    non_cats = list(set(df.columns) - set(cats))
    return cats, non_cats
def aggregate_heuristics(df, grouper):
    cats_, non_cats_ = find_cats_non_cats(df, grouper)
    cats = cats_[:] + grouper
    non_cats = non_cats_[:] + grouper
    
    if not cats_:
        return (df.groupby(grouper).size().to_frame("size").
               merge(df[non_cats].groupby(grouper).mean(),
               left_index = True, right_index = True))
    if not non_cats_:
        return (df.groupby(grouper).size().to_frame("size").
               merge(df[cats].groupby(grouper).agg(lambda x: scipy.stats.mode(x)[0]),
               left_index = True, right_index = True))
    else:
        return (df.groupby(grouper).size().to_frame("size").
               merge(df[cats].groupby(grouper).agg(lambda x: scipy.stats.mode(x)[0]),
               left_index = True, right_index = True).
               merge(df[non_cats].groupby(grouper).mean(),
                    left_index = True, right_index = True))
