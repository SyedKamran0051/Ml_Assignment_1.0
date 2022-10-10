import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Transformers.lagtransformer import Lags
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import Row
import pytest
from pyspark.sql.types import *


@pytest.fixture(scope="session")
def wallmart_data(spark):
    schema = StructType([
        StructField("store_id", StringType()),
        StructField("dept_id", StringType()),
        StructField("year", IntegerType()),
        StructField("month", IntegerType()),
        StructField("sales", DoubleType()),
        StructField("sell_price", DoubleType())
    ])

    data = [
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 1, "sales": 120.0, "sell_price": 152.25},
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 2, "sales": 320.0, "sell_price": 252.25},
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 3, "sales": 430.0, "sell_price": 352.25},
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 4, "sales": 540.0, "sell_price": 752.25},
        {"store_id": "TX_1", "dept_id": "HOBBIES_2", "year": 2015,"month": 1, "sales": 630.0, "sell_price": 352.25},
        {"store_id": "TX_1", "dept_id": "HOBBIES_2", "year": 2015,"month": 2, "sales": 740.0, "sell_price": 752.25}
    ]

    df = spark.createDataFrame([Row(**x) for x in data], schema)
    df.cache()
    df.count()
    return df

def test_log_is_correct(spark, wallmart_data):
    schema = StructType([
        StructField("store_id", StringType()),
        StructField("dept_id", StringType()),
        StructField("year", IntegerType()),
        StructField("month", IntegerType()),
        StructField("sales", DoubleType()),
        StructField("sell_price", DoubleType()),
        StructField("lag_1", DoubleType()),
        StructField("lag_2", DoubleType()),
        StructField("lag_3", DoubleType())
    ])

    data = [
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 1, "sales": 120.0, "sell_price": 152.25, "lag_1": None, "lag_2": None, "lag_3": None},
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 2, "sales": 320.0, "sell_price": 252.25, "lag_1": 120.0, "lag_2": None, "lag_3": None},
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 3, "sales": 430.0, "sell_price": 352.25, "lag_1": 320.0, "lag_2": 120.0, "lag_3": None},
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 4, "sales": 540.0, "sell_price": 752.25, "lag_1": 430.0, "lag_2": 320.0, "lag_3": 120.0},
        {"store_id": "TX_1", "dept_id": "HOBBIES_2", "year": 2015,"month": 1, "sales": 630.0, "sell_price": 352.25, "lag_1": None, "lag_2": None, "lag_3": None},
        {"store_id": "TX_1", "dept_id": "HOBBIES_2", "year": 2015,"month": 2, "sales": 740.0, "sell_price": 752.25, "lag_1": 630.0, "lag_2": None, "lag_3": None}
    ]

    expected_result = spark.createDataFrame([Row(**x) for x in data], schema)
    lag_feature_transform = Lags(lags=[1,2,3], target="sales", partitionBy=["store_id","dept_id"], orderBy=["year", "month"])
    result = lag_feature_transform.transform(wallmart_data)
    assert_df_equality(result, expected_result)
