from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols
from pyspark.ml.param import Param, Params
from pyspark import keyword_only
import pyspark.sql.functions as F
import pandas as pd
import os, sys
import findspark
from pyspark.sql import SparkSession
from pyspark.ml import Model
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Preprocessing.data_manipulation import DataPreparation
from Evaluator.Mape import MAPE


class XGBoostModel(Model, HasLabelCol, HasInputCols, HasPredictionCol):
    findspark.init()
    model = Param(
        Params._dummy(),
        "model",
        "model",
        None,
    )

    @keyword_only
    def __init__(self, labelCol=None, inputCols=None, predictionCol=None, model=None):
        self.spark = SparkSession.builder.getOrCreate()
        super().__init__()
        self._setDefault(model=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol=None, inputCols=None, predictionCol=None, model=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getModel(self):
        return self.getOrDefault(self.model)

    def _transform(self, df):
        if not self.isSet("inputCols"):
            raise ValueError(
                "No input columns set for the "
                "XGBOOST MODEL")

        featureCols = self.getInputCols()
        labelCol = self.getLabelCol()
        pred = self.getPredictionCol()
        model = self.getModel()

        X = df[featureCols].toPandas()
        y = df.select(labelCol).toPandas()

        prediction = model.predict(X)
        resultDf = pd.DataFrame({"store": X["store_id_index"], "year": X["year_index"], "month": X["month"],
                                 pred: prediction, 'actual': y[labelCol]})
        result = self.spark.createDataFrame(resultDf)
        result.createOrReplaceTempView('result')
        return result
