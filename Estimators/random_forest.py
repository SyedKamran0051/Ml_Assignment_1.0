from pyspark.ml import Estimator
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasFeaturesCol
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only
import pyspark.sql.functions as F
from pyspark.ml import Model
from pyspark.ml.regression import RandomForestRegressor
from Evaluator.Mape import MAPE

from Preprocessing.data_manipulation import DataPreparation

class RandomForestClassifier(Estimator, HasPredictionCol, HasLabelCol, HasFeaturesCol):
    
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

        if params is None:
            rf = RandomForestRegressor(featuresCol=features, labelCol=labels)
        else:
            rf = RandomForestRegressor(featuresCol=features, labelCol=labels, **params)

        rf = rf.fit(train)
        predictions = rf.transform(validation)
        mape = MAPE(labelCol="sales", predictionCol=self.getPredictionCol())
        score = mape.evaluate(predictions)
        print("score:", score)
        return {'loss': score, 'model': rf}
    
    def _fit(self, df):
        features = self.getFeaturesCol()
        labels = self.getLabelCol()

        df_1 = DataPreparation()
        train, test = df_1.train_test_split(df_1, 2015)

        self.trainModel(train, test, None)
        rf = RandomForestRegressor(featuresCol=features, labelCol=labels)
        return rf.fit(train)

