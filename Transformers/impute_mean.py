###############################
##### Average Sell Price ######
###############################

from pyspark import keyword_only
from pyspark.ml import Transformer


class ImputePrice(Transformer):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        id = "id"
        selling_price ="sell_price"
        aggregating_id = "agg_id"
        avg_selling_price = "avg(sell_price)"

        df_aggregated = df.groupBy(id).avg(selling_price)
        df_aggregated = df_aggregated.withColumnRenamed(id, aggregating_id)
        df = df.join(df_aggregated, df[id] == df_aggregated[aggregating_id], "inner")
        df = df.drop(aggregating_id, selling_price)
        df = df.withColumnRenamed(avg_selling_price, selling_price)
        return df

