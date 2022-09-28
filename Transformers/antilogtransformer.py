######################################################
# Data Aggregation Transformer
#####################################################

from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasInputCols
from pyspark import keyword_only
import pyspark.sql.functions as Function

class antiLog(Transformer, HasInputCols):
    @keyword_only
    def __init__(self, inputCols=None):
        super().__init__()
        self._setDefault()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def setInputCols(self, new_inputCols):
        return self.setParams(inputCols=new_inputCols)

    def _transform(self, df:DataFrame):
        columns = self.getInputCols()
        for column in columns:
            df = df.withColumn(column, Function.exp(df[column]))
        return df
