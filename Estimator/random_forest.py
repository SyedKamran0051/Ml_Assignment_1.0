
from pyspark.ml import Estimator
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasFeaturesCol
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.regression import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pyspark.sql.functions import abs, mean

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Preprocessing.data_manipulation import DataPreparation
from Evaluator.Mape import MAPE
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from functools import partial
import numpy as np
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


class RandomForest(Estimator, HasPredictionCol, HasLabelCol, HasFeaturesCol):

    search_space = {
        'maxDepth': hp.quniform('maxDepth', 1, 20, 1),
        'numTrees': hp.quniform('numTrees', 60, 180, 5),
        'maxBins': hp.quniform('maxBins', 30, 50, 1),
        'subsamplingRate': hp.choice('subsamplingRate', [1, 0.9])
        }
 
    @keyword_only
    def __init__(self, labelCol=None, featuresCol=None, predictionCol=None):
        super().__init__()
        self._setDefault(predictionCol="prediction")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol=None, featuresCol=None, predictionCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def _fit(self, df:DataFrame):
        
        features = self.getFeaturesCol()
        labels = self.getLabelCol()

        data = DataPreparation()
        train, validation = data.train_test_split(df, 2014)
        
        best = self.optimizing(train, validation)
        
        # fitting the training data on best parameters found using hyperopt
        randomforestmodel = RandomForestRegressor(featuresCol=features, labelCol=labels, **best)
        rfmodel = randomforestmodel.fit(train)
        
        return rfmodel


    def hyper_tuning(self, train, validation, params):
        
        features = self.getFeaturesCol()
        labels = self.getLabelCol()
        pred = self.getPredictionCol()
        
        if params is None:
            randomforestmodel = RandomForestRegressor(featuresCol=features, labelCol=labels)
        else:
            randomforestmodel = RandomForestRegressor(featuresCol=features, labelCol=labels, **params)

        randomforestmodel = randomforestmodel.fit(train)
        predictions = randomforestmodel.transform(validation)

        # Mape Evaluator 
        mape = MAPE(labelCol="sales", predictionCol=pred)
        score = mape.evaluate(predictions)
        loss = score
        # printing score
        print("mape score:", score)
    
        return {"loss": loss, "status": STATUS_OK, "model": randomforestmodel}
    
    def optimizing(self, train, validation):#calling hyper_tuning 

        self.hyper_tuning(train, validation, params=None)

         # finding best parameters 
        trials = Trials()
        best = fmin(partial(self.hyper_tuning, train, validation),
                    space=self.search_space,
                    algo=tpe.suggest,
                    max_evals=5,
                    trials=trials)
        best_parameter = space_eval(self.search_space, best)
        print(best_parameter)
        return best_parameter


