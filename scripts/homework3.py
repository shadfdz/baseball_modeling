# HW1
# Author: Shad Fernandez
# Date: 02-OCT-2021
import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession


def main():

    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.jars", "../dbConnectors/mysql-connector-java-8.0.25.jar")
        .getOrCreate()
    )

    table = "(select g.game_id, g.local_date, bc.batter, bc.atBat, bc.Hit from game g join batter_counts bc on g.game_id \
     = bc.game_id) as rolling_lookup"
    batter_counts_df = (
        spark.read.format("jdbc")
        .option(
            "url",
            "jdbc:mysql://localhost:3306/baseball?zeroDateTimeBehavior=convertToNull",
        )
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .option("dbtable", table)
        .option("user", "guest")
        .option("password", "squidgames")
        .load()
    )

    batter_counts_df.createOrReplaceTempView("batter_counts")
    batter_counts_df.persist(StorageLevel.DISK_ONLY)

    rolling_ave_query = "Select r11.batter, r11.game_id, r11.local_date,  \
                        round(sum(r12.Hit)/sum(r12.atBat),4) as batting_ave from \
                        (select game_id, local_date, batter, atBat, Hit from batter_counts \
                        where atBat > 0 order by batter, local_date) r11 \
                        join (select game_id, local_date, batter, atBat, Hit from batter_counts \
                        where atBat > 0 order by batter, local_date) r12 on r11.batter = r12.batter \
                        and r12.local_date between date_sub(r11.local_date, 100) and r11.local_date \
                        group by r11.batter, r11.game_id, r11.local_date order by batter, local_date"

    result = spark.sql(rolling_ave_query)
    result.show()


if __name__ == "__main__":
    sys.exit(main())
