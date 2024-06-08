import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Evaluator.Mape import MAPE
from pyspark.sql import Row
import pytest
from pyspark.sql.types import *
import unittest

@pytest.fixture(scope="session")
def wallmart_data(spark):
    schema = StructType([

        StructField("prediction", DoubleType()),
        StructField("actual", DoubleType())
    ])

    data = [
        {"prediction": 152.25, "actual": 130.25},
        {"prediction": 252.25, "actual": 250.35},
        {"prediction": 352.25, "actual": 400.50},
        {"prediction": 752.25, "actual": 800.50},
        {"prediction": 352.25, "actual": 405.50},
        {"prediction": 752.25, "actual": 752.50},
    ]

    df = spark.createDataFrame([Row(**x) for x in data], schema)
    df.cache()
    df.count()
    return df

def test_log_case_correct(spark, wallmart_data):
    expected_result = 0.081
    Evaluator_mape = MAPE(predictionCol="prediction", labelCol="actual")
    evaluation = Evaluator_mape.evaluate(wallmart_data)
    evaluation = round(evaluation, 3)
    assert evaluation == expected_result
