###########################################
####### Test_Train_Split Transformer ######
###########################################

from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only

class Split(Transformer):

    year = Param(
        Params._dummy(),
        "split_value",
        "Column remove negative sales from",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, year=None):
        super().__init__()
        self._setDefault()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, year=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getsplitvalue(self):
        return self.getOrDefault(self.year)

    def _transform(self, df, year=2015):
        year = self.getsplitvalue()
        df_train, df_test = df[df['year']<year], df[df['year']>=year]
        return df_train, df_test


