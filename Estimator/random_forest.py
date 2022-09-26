from pyspark.ml import Estimator
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasFeaturesCol
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only
import pyspark.sql.functions as F
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

    def trainModel(self, train, validation, params):
        features = self.getFeaturesCol()
        labels = self.getLabelCol()
    
        rf = RandomForestRegressor(featuresCol=features, labelCol=labels)
        rf = rf.fit(train)
        predictions = rf.transform(validation)
        return predictions
    
    def _fit(self, df):
        features = self.getFeaturesCol()
        labels = self.getLabelCol()

        df_1 = DataPreparation()
        train_df, validation_df = df_1.train_test_split(df_1, 2015)

        self.trainModel(train_df, validation_df, None)
        rf = RandomForestRegressor(featuresCol=features, labelCol=labels)
        return rf.fit(train_df)

