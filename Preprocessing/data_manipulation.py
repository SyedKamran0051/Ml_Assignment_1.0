from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from typing import Iterable


class DataPreparation:
    def __init__(self):
        self.spark = SparkSession.builder.getOrCreate()

    def read_data(self):
        calender = self.spark.read \
            .format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("D:\Data_M5\calendar.csv")

        train_data = self.spark.read.csv("D:\Data_M5\train_data_modified.csv", header=True, inferSchema=True)
        sell_price = self.spark.read.csv("D:\Data_M5\sell_prices.csv", header=True, inferSchema=True)

        return calender, train_data, sell_price

    def get_data(self):
        calendar, train_data, sell_price = self.read_data()
        df = train_data.join(calendar, calendar.d == train_data.d, "left")
        df = df.drop("d")
        df = df.join(sell_price, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        return df
    
    @staticmethod
    def filter_store(df, store_name):
        return df.filter(df.store_id == store_name)
        






