import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Transformers.data_aggregation import AggregateData
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
        StructField("sell_price", DoubleType()),
        StructField("snapWI", LongType())
    ])

    data = [
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 1, "sales": 120.0, "sell_price": 152.25, "snapWI": 1300},
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 2, "sales": 320.0, "sell_price": 252.25, "snapWI": 2300},
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 3, "sales": 430.0, "sell_price": 352.25, "snapWI": 3300},
        {"store_id": "TX_1", "dept_id": "HOBBIES_2", "year": 2014,"month": 1, "sales": 540.0, "sell_price": 752.25, "snapWI": 4300},
        {"store_id": "TX_1", "dept_id": "HOBBIES_2", "year": 2015,"month": 2, "sales": 630.0, "sell_price": 352.25, "snapWI": 5300},
        {"store_id": "TX_1", "dept_id": "HOBBIES_2", "year": 2015,"month": 4, "sales": 740.0, "sell_price": 752.25, "snapWI": 6300},
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
        StructField("sell_price", DoubleType()),
        StructField("snapWI", LongType()),
        StructField("sales", DoubleType())
    ])

    data = [
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 1, "sell_price": 152.25, "snapWI": 1300, "sales": 120.0,},
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 2, "sell_price": 252.25, "snapWI": 2300, "sales": 320.0},
        {"store_id": "CA_1", "dept_id": "HOBBIES_1", "year": 2014,"month": 3, "sell_price": 352.25, "snapWI": 3300, "sales": 430.0,},
        {"store_id": "TX_1", "dept_id": "HOBBIES_2", "year": 2014,"month": 1, "sell_price": 752.25, "snapWI": 4300,  "sales": 540.0},
        {"store_id": "TX_1", "dept_id": "HOBBIES_2", "year": 2015,"month": 2, "sell_price": 352.25, "snapWI": 5300, "sales": 630.0},
        {"store_id": "TX_1", "dept_id": "HOBBIES_2", "year": 2015,"month": 4, "sell_price": 752.25, "snapWI": 6300, "sales": 740.0,},
    ]

    expected_result = spark.createDataFrame([Row(**x) for x in data], schema)
    aggregate = AggregateData(columns=["store_id", "dept_id", "year", "month"],
                                expressions={"sales": "sum",
                                "sell_price": "avg",
                                "snapWI": "sum",
                            })
    result = aggregate.transform(wallmart_data)
    assert_df_equality(result, expected_result)