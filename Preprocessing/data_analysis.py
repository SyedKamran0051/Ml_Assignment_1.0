from pyspark.sql import SparkSession
from data_manipulation import DataPreparation


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("project_spark") \
        .config("spark.driver.memory", "8g")\
        .getOrCreate()

    calender = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("D:\Spark_project\calendar.csv")

    calender.show()
    calender.printSchema()
    train_data = spark.read.csv("D:\\Spark_project\\train_data_modified.csv", header=True, inferSchema=True)
    sell_price = spark.read.csv("D:\Spark_project\sell_prices.csv", header=True, inferSchema=True)
    submission_file = spark.read.csv("D:\Spark_project\sample_submission.csv", header=True, inferSchema=True)

    train_data.show()
    sell_price.show()
    submission_file.show()

    data = DataPreparation()
    df = data.get_data()

    df.show()









