#########################################
##### Tranformer to fill Null Values ####
#########################################


from multiprocessing.sharedctypes import Value
from re import S
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only
import pyspark.sql.functions as F

class ScallerNAFiller(Transformer):
    @keyword_only
    def __init__(self):
        super().__init__()
        self._setDefault()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def _transform(self, df:DataFrame):
        df = df.fillna(value=0)
        return df
