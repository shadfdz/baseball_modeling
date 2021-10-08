from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols


class BatAveRollingAveNDaysTransform(Transformer, HasInputCols):
    @keyword_only
    def __init__(self, inputCols=None):
        super(BatAveRollingAveNDaysTransform, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, spark):

        days = self.getInputCols()

        t_rolling_ave_query = "CREATE TEMPORARY VIEW t_rolling_ave AS \
                              SELECT g.game_id, local_date, batter, atBat, Hit \
                              FROM batter_counts bc join game g on g.game_id = bc.game_id \
                              WHERE atBat > 0"

        spark.sql(t_rolling_ave_query)

        rolling_ave_query = (
            "CREATE TEMPORARY VIEW rolling_ave_100 AS \
                            SELECT  r11.batter, r11.game_id, r11.local_date, \
                            ROUND(SUM(r12.Hit)/SUM(r12.atBat),4) AS BattingAve \
                            FROM t_rolling_ave r11 JOIN t_rolling_ave r12 ON \
                            r11.batter = r12.batter AND r12.local_date BETWEEN \
                            DATE_SUB(r11.local_date, "
            + days[0]
            + ") AND r11.local_date \
                            GROUP BY r11.batter, r11.game_id, r11.local_date"
        )

        spark.sql(rolling_ave_query)

        dataset = spark.sql("SELECT * FROM rolling_ave_100")

        return dataset