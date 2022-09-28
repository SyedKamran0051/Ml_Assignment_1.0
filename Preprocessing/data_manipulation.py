from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from typing import Iterable
import yaml
import os
import pandas as pd

# folder to load config file
CONFIG_PATH = "./config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("config.yaml")

# load data
data_train_path = os.path.join(config["data_directory"], config["training_path"])
data_calender_path = os.path.join(config["data_directory"], config["calender_path"])
data_sell_price_path = os.path.join(config["data_directory"], config["sell_price_path"])
print(data_calender_path)

class DataPreparation:

    def __init__(self):
        self.spark = SparkSession.builder.getOrCreate()

    def read_data(self):
        calender = self.spark.read \
            .format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load(data_calender_path) 

        train_data = self.spark.read.csv(data_train_path, header=True, inferSchema=True)
        sell_price = self.spark.read.csv(data_sell_price_path, header=True, inferSchema=True)
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
    @classmethod
    def train_test_split(cls, df, year=2016):
        train_df, test_df = df[df['year'] < year], df[df['year'] >= year]
        return train_df, test_df
