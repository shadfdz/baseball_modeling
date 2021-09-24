/*
Homework 2
Shad Fernandez
Date: 19-SEP-2021
*/

-- double check numbers, null values, and rounding rules
-- use baseball;

-- Annual RBI? by each player
SELECT
	YEAR(bg.local_date) AS 'Year',
	bc.batter,
	bc.atBat,
    bc.Hit,
    AVG(bc.Hit / bc.atBat) AS 'Batting Average'
FROM baseball.game bg
LEFT JOIN batter_counts bc on bg.game_id = bc.game_id
GROUP BY bg.local_date, bc.batter, bc.Hit, bc.atBat
Limit 10;


-- Historical RBI by each player
SELECT 
	bc.batter,
	-- bc.atBat,
    -- bc.Hit,
    AVG(bc.Hit / bc.atBat) AS 'Batting Average' 
FROM baseball.game bg
LEFT JOIN batter_counts bc on bg.game_id = bc.game_id
GROUP BY bg.local_date, bc.batter, bc.Hit, bc.atBat
    limit 10;

-- Find min date

# 100 day moving average for each player
# WITH RECURSIVE cte (n) AS
# (
#   SELECT 1
#   UNION ALL
#   SELECT n + 1 FROM cte WHERE n < 5
# )
# SELECT * FROM cte;

-- Using recursive to create dates


DROP TEMPORARY TABLE if exists DateRange;

SET SESSION cte_max_recursion_depth = 1000000;
CREATE TEMPORARY TABLE DateRange
WITH RECURSIVE DateRange (DateName) AS
(
  SELECT (select date(min(local_date)) from game)
  UNION ALL
  SELECT adddate(DateName, 1) FROM DateRange WHERE DateName < '2008-12-26'
)
SELECT
* FROM DateRange
limit 10;



-- this is it
SELECT
    DATE(intbl.local_date),
    intbl.batter,
    intbl.BattingAverage,
    game_id,
    avg(ifnull(intbl.BattingAverage, 0)) over (partition by batter order by local_date asc range interval 3 day preceding interval '1' day) AS 'RolAve'
FROM ( SELECT
	bg.local_date,
    bc.game_id,
    batter,
    ifnull(bc.Hit / bc.atBat,0) AS 'BattingAverage'
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
order by batter, local_date) intbl
group by local_date, batter, BattingAverage, game_id
    limit 1000;

