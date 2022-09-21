from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasInputCols, HasPredictionCol
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only
import pyspark.sql.functions as F
from pyspark.ml import Model
from pyspark.ml.classification import DecisionTreeClassifier,GBTClassifier,RandomForestClassifier

class RandomForestClassifier(Transformer, HasPredictionCol):
    @keyword_only
    def __init__(self, PredictionCol=None , Models=None):
        super().__init__()
        self._setDefault(PredictionCol=None, Models=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, PredictionCol=None, Models=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def setPredictionCol(self, newPredictionCol):
        return self.setParams(PredictionCol=newPredictionCol)

    def setModels(self, new_Models):
        return self.setParams(Models=new_Models)
    
    def getModels(self):
        return self.getOrDefault(self.Models)

    def _transform(self, df:DataFrame):
        oredictionCol = self.getPredictionCol
        models = self.getModels
        df = df.sel

        return df