from pyspark.sql import SparkSession
from data_manipulation import DataPreparation


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("project_spark") \
        .config("spark.driver.memory", "8g")\
        .getOrCreate()

    data = DataPreparation()
    df = data.get_data()

    df.show()













