from ast import IsNot
from pyspark.ml import Estimator, param
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only
import xgboost as xgb
import pandas as pd
import numpy as np
from functools import partial
from sklearn.metrics import mean_squared_error, roc_auc_score

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Preprocessing.data_manipulation import DataPreparation
from Evaluator.Mape import MAPE
from Models.xgboost import XGBoostModel

# Hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe

class XGBoost(Estimator, HasPredictionCol, HasLabelCol, HasInputCols):

    # define hyperopt search space
    search_space ={
    'max_depth':  hp.choice('max_depth', np.arange(1, 30, dtype=int))                                 # depth of trees (preference is for shallow trees or even stumps (max_depth=1))
    ,'learning_rate' : hp.loguniform('learning_rate', np.log(0.05), np.log(0.40))     # learning rate for XGBoost
    ,'gamma': hp.quniform('gamma', 0.0, 1.0, 0.001)                                   # minimum loss reduction required to make a further partition on a leaf node
    ,'min_child_weight' : hp.quniform('min_child_weight', 1, 30, 1)                   # minimum number of instances per node
    ,'subsample' : hp.loguniform('subsample', np.log(0.1), np.log(1.0))               # random selection of rows for training,
    ,'colsample_bytree' : hp.loguniform('colsample_bytree', np.log(0.1), np.log(1.0)) # proportion of columns to use per tree
    ,'colsample_bylevel': hp.loguniform('colsample_bylevel', np.log(0.1), np.log(1.0))# proportion of columns to use per level
    ,'colsample_bynode' : hp.loguniform('colsample_bynode', np.log(0.1), np.log(1.0)) # proportion of columns to use per node
    }
    @keyword_only
    def __init__(self, labelCol=None, inputCols=None, predictionCol=None):
        super().__init__()
        self._setDefault(predictionCol="prediction")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol=None, inputCols=None, predictionCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    

    def hyper_parameter_tuning(self, X_train, y_train, X_val, y_val, params):
        
        print("training for finding best hyper parameters")
        # print training with parameters 
        XGB = xgb.XGBRegressor(**params)
        XGB = XGB.fit(X_train, y_train)
        predictions = XGB.predict(X_val)
        
        # predictions using mape 
        score = np.mean(np.abs((y_val.sales - predictions) / y_val.sales))

        # score = roc_auc_score(y_val, pred)
        print ("mape score:", score)
        loss = score

        return {'loss': loss, 'status': STATUS_OK, 'model': XGB}


    def _fit(self, df):

        input_columns = self.getInputCols()
        labels = self.getLabelCol()
        pred = self.getPredictionCol()

        data = DataPreparation()
        train, validation = data.train_test_split(df, 2014)

        X_train = train.select(input_columns).toPandas()
        y_train = train.select(labels).toPandas()
        X_val = validation.select(input_columns).toPandas()
        y_val = validation.select(labels).toPandas()

        # best parameters to train the model
        self.hyper_parameter_tuning(X_train, y_train, X_val, y_val, params=self.search_space)
        
        trials = Trials()
        best = fmin(partial(self.hyper_parameter_tuning, X_train, y_train, X_val, y_val),
                    space=self.search_space,
                    algo=tpe.suggest,
                    max_evals=10,
                    trials=trials,
                    verbose=True)
        
        best_parameters = space_eval(self.search_space, best)
        # best parameters to train the model
        print(best_parameters)

        # training the input data set with the beet parameters 
        df_X = df[input_columns].toPandas()
        df_y = df.select(labels).toPandas()

        # training data using the best parameters found using hyperopt
        XGB = xgb.XGBRegressor(**best_parameters)
        XGB = XGB.fit(df_X, df_y)

        return XGBoostModel(labelCol=labels, inputCols=input_columns, predictionCol=pred, model=XGB)