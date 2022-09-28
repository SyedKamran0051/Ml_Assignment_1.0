from pyspark.ml import Estimator
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only
import xgboost as xgb
import pandas as pd
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Preprocessing.data_manipulation import DataPreparation
from Evaluator.Mape import MAPE
from Models.xgboost import XGBoostModel


class XGBoost(Estimator, HasPredictionCol, HasLabelCol, HasInputCols):

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

    def _fit(self, df):
        input_columns = self.getInputCols()
        labels = self.getLabelCol()
        pred = self.getPredictionCol()

        XGB = xgb.XGBRegressor(eta=1)
        data = DataPreparation()
        train, validation = data.train_test_split(df, 2015)

        X_train = train.select(input_columns).toPandas()
        y_train = train.select(labels).toPandas()
        X_val = validation.select(input_columns).toPandas()
        y_val = validation.select(labels).toPandas()

        XGB = XGB.fit(X_train, y_train)
        prediction = XGB.predict(X_val)

        return XGBoostModel(labelCol=labels, inputCols=input_columns, predictionCol=pred, model=XGB)