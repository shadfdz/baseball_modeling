# HW1
# Author: Shad Fernandez
# Date: 02-OCT-2021
import sys
from pyspark import StorageLevel
from pyspark.sql import SparkSession

def main():

    spark = SparkSession.builder.master('local[*]') \
    .config("spark.jars","../dbConnectors/mysql-connector-java-8.0.25.jar") \
    .getOrCreate()

    table = '(select g.game_id, g.local_date, bc.batter, bc.atBat bc.Hit from game g join batter_counts bc on g.game_id = bc.game_id) as rolling_lookup'
    batter_counts_df = spark.read.format("jdbc").option("url","jdbc:mysql://localhost:3306/baseball?zeroDateTimeBehavior=convertToNull") \
    .option("driver","com.mysql.cj.jdbc.Driver") \
    .option("dbtable",table).option("user","guest").option("password","squidgames").load()

    batter_counts_df.createOrReplaceTempView("rolling_ave_lookup")
    batter_counts_df.persist(StorageLevel.DISK_ONLY)

    result = spark.sql("SELECT * from rolling_ave_lookup")
    result.show()


if __name__ == "__main__":
    sys.exit(main())
