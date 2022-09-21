from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only

class Split(Transformer):
    
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super().__init__()
        self._setDefault()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def _transform(self, df):
        df_train, df_test = self.df.randomsplit([0.7,0.3])
        return df_train, df_test

