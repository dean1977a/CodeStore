param = {'max_depth': 20,
         'learning_rate': 0.1,
         'objective': 'regression',
         'boosting_type': 'gbdt',
         'verbose': 1,
         'metric': 'rmse',
         'seed': 42,
         'n_jobs': 12}

original_columns = ['IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName','ExitStreetName', 'EntryHeading', 
                        'ExitHeading', 'Hour', 'Weekend', 'Month', 'City', 'EntryType', 'ExitType']
def forwardFeatureSelect(model,df,original_columns,target):
    train_columns = list(train.columns[13:])
    usefull_columns = []
    not_usefull_columns = []
    best_score = 0
    
    train_tmp = train[original_columns]
    print('Training with {} features'.format(train_tmp.shape[1]))
    x_train, x_val, y_train, y_val = train_test_split(train_tmp, target, test_size = 0.2, random_state = 42)
    xg_train = lgb.Dataset(x_train, label = y_train)
    xg_valid = lgb.Dataset(x_val, label= y_val)
    clf = lgb.train(param, xg_train, 100000, valid_sets = [xg_train, xg_valid], verbose_eval = 3000, 
                    early_stopping_rounds = 100)
    predictions = clf.predict(x_val)
    rmse_score = np.sqrt(mean_squared_error(y_val, predictions))
    print("RMSE baseline val score: ", rmse_score)
    best_score = rmse_score
    
    for num, i in enumerate(train_columns):
        train_tmp = train[original_columns + usefull_columns + [i]]
        print('Training with {} features'.format(train_tmp.shape[1]))
        x_train, x_val, y_train, y_val = train_test_split(train_tmp, target, test_size = 0.2, random_state = 42)
        xg_train = lgb.Dataset(x_train, label = y_train)
        xg_valid = lgb.Dataset(x_val, label= y_val)   

        #lightgbm
        if model == 'lightgbm':
            clf = lgb.train(param, xg_train, 100000, valid_sets = [xg_train, xg_valid], verbose_eval = 3000, 
                        early_stopping_rounds = 100)
        #ridge
        if model == 'ridge':
            scaler = MinMaxScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)
            print( x_train.shape, x_val.shape, y_train.shape )
            clf = Ridge(alpha=20, copy_X=True, fit_intercept=True, solver='auto',max_iter=10000,normalize=False, random_state=0,  tol=0.0025)
        
        predictions = clf.predict(x_val)
        rmse_score = np.sqrt(mean_squared_error(y_val, predictions))
        print("RMSE val score: ", rmse_score)
        
        if rmse_score < best_score:
            print('Column {} is usefull'.format(i))
            best_score = rmse_score
            usefull_columns.append(i)
        else:
            print('Column {} is not usefull'.format(i))
            not_usefull_columns.append(i)
            
        print('Best rmse score for iteration {} is {}'.format(num + 1, best_score))
        
    return usefull_columns, not_usefull_columns
            
usefull_columns, not_usefull_columns = forwardFeatureSelect(model,df,original_columns,target)
