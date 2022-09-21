from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only
import pyspark.sql.functions as F

class ScalarNAFiller(Transformer, HasInputCols):
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
        input_columns = self.getInputCols()
        for column in input_columns:
            df = df.withColumn(column, df[column].fillna(6.0))
        return df

        
