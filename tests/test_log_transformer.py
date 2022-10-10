
import pyspark.sql.functions as Function
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Transformers.logtransformer import Log
from chispa.dataframe_comparer import assert_df_equality
from pyspark_test import assert_pyspark_df_equal
from pyspark.sql import Row
import pytest
from pyspark.sql.types import StructType, StructField, DateType, StringType, IntegerType, DoubleType

@pytest.fixture(scope="session")
def stock_data(spark):

    data = [
        {"sales": 5600},
        {"sales": 18750},
    ]
    df = spark.createDataFrame(Row(**x) for x in data)
    df.cache()
    df.count()
    return df

def test_log_is_correct(spark, stock_data):

    expected_data = [
        {"sales": 8.631},
        {"sales": 9.839},
    ]
    expected_result = spark.createDataFrame(Row(**x) for x in expected_data)
    log_tranformer = Log(inputCols=["sales"])
    result = log_tranformer.transform(stock_data)
    result = result.withColumn("sales", Function.round("sales",3))
    assert_df_equality(result, expected_result)
