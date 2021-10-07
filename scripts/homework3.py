# HW1
# Author: Shad Fernandez
# Date: 02-OCT-2021
import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession


def get_spark(driver_path):
    """
    function creates a spark session and connects to db using a connector
    :param driver_path: file path of db connector
    :return: spark instance
    """
    return (
        SparkSession.builder.master("local[*]")
        .config("spark.jars", driver_path)
        .getOrCreate()
    )


def load_table_spark(spark_instance, url, driver, dtable, user, password):
    """
    Retrieve table from db and load to spark
    :param spark_instance: spark instance
    :param url: url of host
    :param driver: tbe db Driver
    :param dtable: name of table
    :param user: username
    :param password: password
    :return: dataframe
    """
    data_frame = (
        spark_instance.read.format("jdbc")
        .option(
            "url",
            url,
        )
        .option("driver", driver)
        .option("dbtable", dtable)
        .option("user", user)
        .option("password", password)
        .load()
    )

    data_frame.createOrReplaceTempView("batter_counts")

    return data_frame


def main():

    connector_file_path = "../dbConnectors/mysql-connector-java-8.0.25.jar"
    spark = get_spark(connector_file_path)
    game_batter_counts_table = "(select g.game_id, g.local_date, bc.batter, bc.atBat, bc.Hit from game g join batter_counts bc on g.game_id \
     = bc.game_id) as rolling_lookup"

    url = "jdbc:mysql://localhost:3306/baseball?zeroDateTimeBehavior=convertToNull"
    driver = "com.mysql.cj.jdbc.Driver"
    user = "guest"
    password = "squidgames"
    view_name = "batter_counts"
    batter_counts = load_table_spark(
        spark, url, driver, game_batter_counts_table, user, password, view_name
    )
    batter_counts.persist(StorageLevel.DISK_ONLY)


    rolling_ave_query = "Select r11.batter, r11.game_id, r11.local_date,  \
                        round(sum(r12.Hit)/sum(r12.atBat),4) as batting_ave from \
                        (select game_id, local_date, batter, atBat, Hit from batter_counts \
                        where atBat > 0 order by batter, local_date) r11 \
                        join (select game_id, local_date, batter, atBat, Hit from batter_counts \
                        where atBat > 0 order by batter, local_date) r12 on r11.batter = r12.batter \
                        and r12.local_date between date_sub(r11.local_date, 100) and r11.local_date \
                        group by r11.batter, r11.game_id, r11.local_date order by batter, local_date"

    rolling_ave_df = spark.sql(rolling_ave_query)
    rolling_ave_df.show()


if __name__ == "__main__":
    sys.exit(main())
