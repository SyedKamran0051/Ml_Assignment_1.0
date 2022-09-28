######################################################
# Data Aggregation Transformer
#####################################################

from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark import keyword_only


class AggregateData(Transformer):
    columns = Param(
        Params._dummy(),
        "columns",
        "Columns to group by on",
        typeConverter=TypeConverters.toListString,
    )

    expressions = Param(
        Params._dummy(),
        "expressions",
        "Dictionary of aggregate expressions",
        None,
    )

# when you don't know how many keyword arguments are going to pass through your function
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, columns=None, expressions=None):
        super().__init__()
        self._setDefault(columns=None, expressions=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def getColumns(self):
        return self.getOrDefault(self.columns)

    def getExpression(self):
        return self.getOrDefault(self.expressions)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, columns=None, expressions=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        group_by_column = self.getColumns()
        aggregate_expression = dict(self.getExpression())
        df_agg = df.groupBy(group_by_column).agg(aggregate_expression)
        # for renaming the column name after aggregation
        for key, opr in aggregate_expression.items():
            name = "{}({})".format(opr, key)
            df_agg = df_agg.withColumnRenamed(name, key)
        return df_agg
