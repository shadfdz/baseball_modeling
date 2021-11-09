# HW1
# Author: Shad Fernandez
# Date: 02-OCT-2021
import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from rolling_ave_transform import BatAveRollingAveNDaysTransform


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


def load_table_spark(spark_instance, url, driver, table_list, user, passkey):
    """
    Retrieve table(s) from db and load to spark and returns a dictionary of df
    :param table_list:
    :param spark_instance: spark instance
    :param url: url of host
    :param driver: tbe db Driver
    :param user: username
    :param passkey: passkey
    :return dictionary containing df
    """
    df_dict = {}
    for name in table_list:
        query_table = "(SELECT * FROM " + name + ") AS " + name
        data_frame = (
            spark_instance.read.format("jdbc")
            .option(
                "url",
                url,
            )
            .option("driver", driver)
            .option("dbtable", query_table)
            .option("user", user)
            .option("password", passkey)
            .load()
        )
        data_frame.persist(StorageLevel.DISK_ONLY)
        data_frame.createOrReplaceTempView(name)
        df_dict[name] = data_frame
    return df_dict


def main():
    # Get path to db connector and get spark to start spark session
    connector_file_path = "../dbConnectors/mysql-connector-java-8.0.25.jar"
    spark = get_spark(connector_file_path)

    # get user input
    print("Please enter db user info")
    user = input("username: ")
    passkey = input("Password: ")
    # set variables for parameters to get tables from mysql
    url = "jdbc:mysql://localhost:3306/baseball?zeroDateTimeBehavior=convertToNull"
    driver = "com.mysql.cj.jdbc.Driver"
    table_list = ["game", "batter_counts"]

    # retrieve tables and load to spark
    table_dict = load_table_spark(spark, url, driver, table_list, user, passkey)

    # show schema of selected tables
    for obj in table_dict.values():
        obj.printSchema()

    # use transformer to retrieve df with n days of batting ave rolling ave
    # set at 100 days for this examples
    rolling_ave_transform = BatAveRollingAveNDaysTransform(inputCols=["100"])
    rolling_ave_100_days_df = rolling_ave_transform.transform(spark)

    # show dataframe
    rolling_ave_100_days_df.show()


if __name__ == "__main__":
    sys.exit(main())
