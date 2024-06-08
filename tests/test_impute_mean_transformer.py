import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Transformers.impute_mean import ImputePrice
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import Row
import pytest
from pyspark.sql.types import *


@pytest.fixture(scope="session")
def wallmart_data(spark):
    schema = StructType([
        StructField("id", StringType()),
        StructField("dept_id", StringType()),
        StructField("year", IntegerType()),
        StructField("month", IntegerType()),
        StructField("sales", DoubleType()),
        StructField("sell_price", DoubleType())
    ])

    data = [
        {"id": "FOODS_1_001_CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 1, "sales": 120.0, "sell_price": 152.25},
        {"id": "FOODS_1_001_CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 2, "sales": 320.0, "sell_price": 252.25},
        {"id": "FOODS_1_001_CA_2", "dept_id": "HOBBIES_1", "year": 2014,"month": 1, "sales": 430.0, "sell_price": 352.25},
        {"id": "FOODS_1_001_CA_2", "dept_id": "HOBBIES_1", "year": 2014,"month": 2, "sales": 540.0, "sell_price": 752.25},
    ]

    df = spark.createDataFrame([Row(**x) for x in data], schema)
    df.cache()
    df.count()
    return df

def test_log_is_correct(spark, wallmart_data):
    schema = StructType([
        StructField("id", StringType()),
        StructField("dept_id", StringType()),
        StructField("year", IntegerType()),
        StructField("month", IntegerType()),
        StructField("sales", DoubleType()),
        StructField("sell_price", DoubleType())
    ])

    data = [
        {"id": "FOODS_1_001_CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 1, "sales": 120.0, "sell_price": 202.25},
        {"id": "FOODS_1_001_CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 2, "sales": 320.0, "sell_price": 202.25},
        {"id": "FOODS_1_001_CA_2", "dept_id": "HOBBIES_1", "year": 2014,"month": 1, "sales": 430.0, "sell_price": 552.25},
        {"id": "FOODS_1_001_CA_2", "dept_id": "HOBBIES_1", "year": 2014,"month": 2, "sales": 540.0, "sell_price": 552.25},
    ]

    expected_result = spark.createDataFrame((Row(**x) for x in data), schema)
    Impute_avg_price_transformer = ImputePrice()
    result = Impute_avg_price_transformer.transform(wallmart_data)
    assert_df_equality(result, expected_result)
