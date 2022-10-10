import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Transformers.negative_sales import NegativeSales
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import Row
import pytest
from pyspark.sql.types import StructType, StructField, DateType, StringType, IntegerType, DoubleType

@pytest.fixture(scope="session")
def stock_data(spark):

    data = [
        {"sales": -5600},
        {"sales": 18750},
        {"sales": -460},
        {"sales": 1750},
        {"sales": -600},
        {"sales": 3750},
    ]
    df = spark.createDataFrame(Row(**x) for x in data)
    df.cache()
    df.count()
    return df

def test_log_is_correct(spark, stock_data):

    expected_data = [
        {"sales": 0},
        {"sales": 18750},
        {"sales": 0},
        {"sales": 1750},
        {"sales": 0},
        {"sales": 3750},
    ]
    expected_result = spark.createDataFrame(Row(**x) for x in expected_data)
    Negative_sales_transformer = NegativeSales(column="sales")
    result = Negative_sales_transformer.transform(stock_data)
    assert_df_equality(result, expected_result)
