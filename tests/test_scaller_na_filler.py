from pyspark.sql import Row
import pytest
from pyspark.sql import functions as Function
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql.types import *
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Transformers.scalar_na_filler import ScallerNAFiller


@pytest.fixture(scope="session")
def stock_data(spark):
    schema = StructType([
        StructField("sales", DoubleType(), nullable=True),
        StructField("snap_WI", LongType(), nullable=True),
        StructField("lag_1", DoubleType(), nullable=True),
        StructField("sell_price", DoubleType(), nullable=True),
        StructField("lag_2", DoubleType(), nullable=True),
        StructField("lag_3", DoubleType(), nullable=True)
    ])
    data = [
        {"sales": None, "snap_WI": None, "lag_1": 200.0, "sell_price": 152.25, "lag_2": 150.0, "lag_3": None},
        {"sales": 1528.5, "snap_WI": 1540, "lag_1": None,  "sell_price": None, "lag_2": 120.0, "lag_3": None},
        {"sales": 1254.0, "snap_WI": None, "lag_1": 400.50,  "sell_price": None, "lag_2": 110.0, "lag_3": 123.5},
        {"sales": None, "snap_WI": 2450, "lag_1": None,  "sell_price": 1450.5 , "lag_2": 240.0, "lag_3": 1.345}
    ]
    df = spark.createDataFrame([Row(**x) for x in data], schema)
    df.cache()
    df.count()
    return df

def test_log_is_correct(spark, stock_data):

    schema = StructType([
        StructField("sales", DoubleType()),
        StructField("snap_WI", LongType()),
        StructField("lag_1", DoubleType()),
        StructField("sell_price", DoubleType()),
        StructField("lag_2", DoubleType()),
        StructField("lag_3", DoubleType())
    ])

    data = [
        {"sales": 0.0, "snap_WI": 0, "lag_1": 200.0, "sell_price": 152.25, "lag_2": 150.0, "lag_3": 0.0},
        {"sales": 1528.5, "snap_WI": 1540, "lag_1": 0.0,  "sell_price": 0.0, "lag_2": 120.0, "lag_3": 0.0},
        {"sales": 1254.0, "snap_WI": 0, "lag_1": 400.50,  "sell_price": 0.0, "lag_2": 110.0, "lag_3": 123.5},
        {"sales": 0.0, "snap_WI": 2450, "lag_1": 0.0,  "sell_price": 1450.5, "lag_2": 240.0, "lag_3": 1.345}

    ]
    expected_result = spark.createDataFrame([Row(**x) for x in data], schema)
    ScallerNAtransformer = ScallerNAFiller()
    result = ScallerNAtransformer.transform(stock_data)
    assert_df_equality(result, expected_result, ignore_nullable=True)


