# Define all the groupby transformations
GROUPBY_AGGREGATIONS = [
    
    # V1 - GroupBy Features #
    #########################    
    # Variance in day, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
    # Variance in hour, for ip-app-os
    {'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-day-channel
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'},
    # Count, for ip-day-hour
    {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},        
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'}, 
    
    # V2 - GroupBy Features #
    #########################
    # Average clicks on app by distinct users; is it an app they return to?
    {'groupby': ['app'], 
     'select': 'ip', 
     'agg': lambda x: float(len(x)) / len(x.unique()), 
     'agg_name': 'AvgViewPerDistinct'
    },
    # How popular is the app or channel?
    {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},
    
    # V3 - GroupBy Features                                              #
    # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
    ###################################################################### 
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','day'], 'select': 'hour', 'agg': 'nunique'}, 
    {'groupby': ['ip','app'], 'select': 'os', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'}, 
    {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','device','os'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'}, 
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'}    
]

# Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    
    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    
    # Info
    print("Grouping by {}, and aggregating {} with {}".format(
        spec['groupby'], spec['select'], agg_name
    ))
    
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
    
    # Perform the groupby
    gp = X_train[all_features]. \
        groupby(spec['groupby'])[spec['select']]. \
        agg(spec['agg']). \
        reset_index(). \
        rename(index=str, columns={spec['select']: new_feature})
        
    # Merge back to X_total
    if 'cumcount' == spec['agg']:
        X_train[new_feature] = gp[0].values
    else:
        X_train = X_train.merge(gp, on=spec['groupby'], how='left')
        
     # Clear memory
    del gp
    gc.collect()

X_train.head()
#############################################################################
  NEW_AGGREGATION_RECIPIES = [
            (["CODE_GENDER",
              "NAME_EDUCATION_TYPE"], [("AMT_ANNUITY", "max"),
                                       ("AMT_CREDIT", "max"),
                                       ("EXT_SOURCE_1", "median"),
                                       ("EXT_SOURCE_2", "median"),
                                       ("OWN_CAR_AGE", "max"),
                                       ("OWN_CAR_AGE", "sum"),
                                       ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                       ("NEW_SOURCES_MEAN", "median"),
                                       ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                       ("NEW_SOURCES_PROD", "median"),
                                       ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                       ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                       ("NEW_SOURCES_STD", "median"),
                                       ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                       ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                       ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),

            (["CODE_GENDER",
              "ORGANIZATION_TYPE"], [("AMT_ANNUITY", "median"),
                                     ("AMT_INCOME_TOTAL", "median"),
                                     ("DAYS_REGISTRATION", "median"),
                                     ("EXT_SOURCE_1", "median"),
                                     ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                     ("NEW_SOURCES_MEAN", "median"),
                                     ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                     ("NEW_SOURCES_PROD", "median"),
                                     ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                     ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                     ("NEW_SOURCES_STD", "median"),
                                     ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                     ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                     ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),

            (["CODE_GENDER",
              "REG_CITY_NOT_WORK_CITY"], [("AMT_ANNUITY", "median"),
                                          ("CNT_CHILDREN", "median"),
                                          ("DAYS_ID_PUBLISH", "median"),
                                          ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                          ("NEW_SOURCES_MEAN", "median"),
                                          ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                          ("NEW_SOURCES_PROD", "median"),
                                          ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                          ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                          ("NEW_SOURCES_STD", "median"),
                                          ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                          ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                          ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),

            (["CODE_GENDER",
              "NAME_EDUCATION_TYPE",
              "OCCUPATION_TYPE",
              "REG_CITY_NOT_WORK_CITY"], [("EXT_SOURCE_1", "median"),
                                          ("EXT_SOURCE_2", "median"),
                                          ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                          ("NEW_SOURCES_MEAN", "median"),
                                          ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                          ("NEW_SOURCES_PROD", "median"),
                                          ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                          ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                          ("NEW_SOURCES_STD", "median"),
                                          ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                          ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                          ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),
            (["NAME_EDUCATION_TYPE",
              "OCCUPATION_TYPE"], [("AMT_CREDIT", "median"),
                                   ("AMT_REQ_CREDIT_BUREAU_YEAR", "median"),
                                   ("APARTMENTS_AVG", "median"),
                                   ("BASEMENTAREA_AVG", "median"),
                                   ("EXT_SOURCE_1", "median"),
                                   ("EXT_SOURCE_2", "median"),
                                   ("EXT_SOURCE_3", "median"),
                                   ("NONLIVINGAREA_AVG", "median"),
                                   ("OWN_CAR_AGE", "median"),
                                   ("YEARS_BUILD_AVG", "median"),
                                   ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                   ("NEW_SOURCES_MEAN", "median"),
                                   ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                   ("NEW_SOURCES_PROD", "median"),
                                   ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                   ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                   ("NEW_SOURCES_STD", "median"),
                                   ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                   ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                   ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),

            (["NAME_EDUCATION_TYPE",
              "OCCUPATION_TYPE",
              "REG_CITY_NOT_WORK_CITY"], [("ELEVATORS_AVG", "median"),
                                          ("EXT_SOURCE_1", "median"),
                                          ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                          ("NEW_SOURCES_MEAN", "median"),
                                          ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                          ("NEW_SOURCES_PROD", "median"),
                                          ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                          ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                          ("NEW_SOURCES_STD", "median"),
                                          ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                          ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                          ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),

            (["OCCUPATION_TYPE"], [("AMT_ANNUITY", "median"),
                                   ("CNT_CHILDREN", "median"),
                                   ("CNT_FAM_MEMBERS", "median"),
                                   ("DAYS_BIRTH", "median"),
                                   ("DAYS_EMPLOYED", "median"),
                                   ("DAYS_ID_PUBLISH", "median"),
                                   ("DAYS_REGISTRATION", "median"),
                                   ("EXT_SOURCE_1", "median"),
                                   ("EXT_SOURCE_2", "median"),
                                   ("EXT_SOURCE_3", "median"),
                                   ("NEW_CREDIT_TO_ANNUITY_RATIO", "median"),
                                   ("NEW_SOURCES_MEAN", "median"),
                                   ("NEW_CREDIT_TO_GOODS_RATIO", "median"),
                                   ("NEW_SOURCES_PROD", "median"),
                                   ("NEW_CAR_TO_EMPLOY_RATIO", "median"),
                                   ("NEW_PHONE_TO_BIRTH_RATIO", "median"),
                                   ("NEW_SOURCES_STD", "median"),
                                   ("NEW_ANNUITY_TO_INCOME_RATIO", "median"),
                                   ("NEW_EMPLOY_TO_BIRTH_RATIO", "median"),
                                   ("NEW_PHONE_TO_EMPLOY_RATIO", "median")]),
        ]

        for groupby_cols, specs in NEW_AGGREGATION_RECIPIES:
            group_object = self.__application_train.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = "{}_{}_{}_{}".format("NEW", "_".join(groupby_cols), agg, select)
                self.__application_train = self.__application_train.merge(
                    group_object[select]
                        .agg(agg)
                        .reset_index()
                        .rename(index=str, columns={select: groupby_aggregate_name}),
                    left_on=groupby_cols,
                    right_on=groupby_cols,
                    how="left"
                )

