from pyspark.ml import Estimator
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasFeaturesCol
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.regression import RandomForestRegressor

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Preprocessing.data_manipulation import DataPreparation
from Evaluator.Mape import MAPE


class RandomForest(Estimator, HasPredictionCol, HasLabelCol, HasFeaturesCol):

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
        rf = RandomForestRegressor(featuresCol=features, labelCol=labels)
        data = DataPreparation()
        train, validation = data.train_test_split(df, 2015)
        rf = rf.fit(train)
        predictions = rf.transform(validation)
        return predictions