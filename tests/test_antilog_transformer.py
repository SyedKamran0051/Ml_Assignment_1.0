import pyspark.sql.functions as Function
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Transformers.antilogtransformer import AntiLog
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import Row
import pytest
from pyspark.sql.types import StructType, StructField, DateType, StringType, IntegerType, DoubleType, FloatType

@pytest.fixture(scope="session")
def stock_data(spark):
    schema = StructType([
        StructField("sales", DoubleType())
    ])
    data = [
        {"sales": 7.065613363597717},
        {"sales": 7.212294468500341},
        {"sales": 7.355641102974253},

    ]
    df = spark.createDataFrame([Row(**x) for x in data], schema)
    df.cache()
    df.count()
    return df

def test_log_is_correct(spark, stock_data):
    schema = StructType([
        StructField("sales", DoubleType())
    ])
    
    data = [
        {"sales": 1171.0},
        {"sales": 1356.0},
        {"sales": 1565.0},

    ]
    expected_result = spark.createDataFrame([Row(**x) for x in data], schema)
    antilog_tranformer = AntiLog(inputCols=["sales"])
    result = antilog_tranformer.transform(stock_data)
    result = result.withColumn("sales", Function.round("sales", 1))
    assert_df_equality(result, expected_result)
