# HW1
# Author: Shad Fernandez
# Date: 02-OCT-2021
import sys
import tempfile
import requests

import pymysql
from pyspark import StorageLevel
from pyspark.sql import SparkSession

def main():

    spark = SparkSession.builder.master('local[*]') \
    .config("spark.jars","../dbConnectors/mysql-connector-java-8.0.25.jar") \
    .getOrCreate()


    batter_counts_df = spark.read.format("jdbc").option("url","jdbc:mysql://localhost:3306/baseball") \
    .option("driver","com.mysql.cj.jdbc.Driver") \
    .option("dbtable","batter_counts").option("user","root").option("password","sd.xd.mmc").load()

    batter_counts_df.createOrReplaceTempView("batter_counts")
    batter_counts_df.persist(StorageLevel.DISK_ONLY)
    # results = spark.sql("SELECT game_id, batter, atBat, Hit from batter_counts limit 10")
    # results.show()

    spark2 = SparkSession.builder.master('local[*]') \
    .config("spark.jars","../dbConnectors/mysql-connector-java-8.0.25.jar") \
    .getOrCreate()

    game_df = spark2.read.format("jdbc").option("url","jdbc:mysql://localhost:3306/baseball") \
    .option("driver","com.mysql.cj.jdbc.Driver") \
    .option("dbtable","game").option("user","root").option("password","sd.xd.mmc").load()

    game_df.createOrReplaceTempView("game")
    game_df.persist(StorageLevel.DISK_ONLY)


    # results2 = spark2.sql("SELECT game_id, local_date from game limit 20")
    # results2.show()

if __name__ == "__main__":
    sys.exit(main())
