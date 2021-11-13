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




SELECT
	bc.batter,
	-- bc.atBat,
    -- bc.Hit,
       DATE(bg.local_date),
    AVG(bc.Hit / bc.atBat) AS 'Batting Average'
FROM baseball.game bg
LEFT JOIN batter_counts bc on bg.game_id = bc.game_id
WHERE batter = '112128'
GROUP BY bg.local_date, bc.batter, bc.Hit, bc.atBat
    limit 100;

-- -----------------------------------

DROP TEMPORARY TABLE if exists DateRange;
-- set recursion depth to difference in max min date
SET SESSION cte_max_recursion_depth = 1000000;
CREATE TEMPORARY TABLE DateRange
WITH RECURSIVE DateRange (DateName) AS
(
  SELECT (select date(min(local_date)) from game)
  UNION ALL
  SELECT adddate(DateName, 1) FROM DateRange WHERE DateName < '2009-12-26'
)
SELECT DateName
FROM DateRange;

-- # Add dates missing dates to each batter

DROP TEMPORARY TABLE if exists batting_ave;
CREATE TEMPORARY TABLE batting_ave
SELECT
batter,
Hit,
date(local_date) as 'game_date'
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
order by batter, local_date;

DROP TEMPORARY TABLE if exists baseball.batting_ave_2;
CREATE TEMPORARY TABLE batting_ave_2
SELECT
batter,
Hit,
date(local_date) as 'game_date'
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
order by batter, local_date;;


DROP TEMPORARY TABLE if exists baseball.batting_ave_3;
CREATE TEMPORARY TABLE batting_ave_3
SELECT
batter,
Hit,
date(local_date) as 'game_date'
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
order by batter, local_date;



















SELECT DateName,
       g.local_date,
       ifnull(batter, (Select batter from batter_counts where batter = '111213' group by batter)),
       ifnull(bc.atBat,0),
       ifnull(Hit,0),
       ifnull(Hit / bc.atBat,0) as 'RBI'
from DateRange
left outer join game g on Date(g.local_date) = DateName
left outer join (
    Select
    game_id,
    atBat,
    Hit,
    batter
    from batter_counts
    where batter = '111213'
    ) bc on bc.game_id = g.game_id
limit 900;


DROP TEMPORARY TABLE if exists batting_ave;
CREATE TEMPORARY TABLE batting_ave
SELECT
bg.game_id,
batter,
Hit,
atBat,
date(local_date) as 'game_date'
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
order by batter, local_date
limit 20;

DROP TEMPORARY TABLE if exists batting_ave;
CREATE TEMPORARY TABLE batting_ave_2
SELECT
bg.game_id,
batter,
Hit,
atBat,
date(local_date) as 'game_date'
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
order by batter, local_date
limit 20;



Select *
from batting_ave t1
limit 20;




# SELECT date, SUM(close),
#        (select avg(close) from tbl t2 where t2.name_id = t.name_id and datediff(t2.date, t.date) <= 9
#        ) as mvgAvg
# FROM tbl t
# WHERE date <= '2002-07-05' and
#       name_id = 2
# GROUP BY date
# ORDER BY date DESC

SELECT
game_date,
hit,
batter,
       (
           select sum(t2.hit)
           from batting_ave_2 t2
           where t2.batter = t1.batter and t2.game_id = t1.game_id
           and datediff(t2.game_date, t1.game_date) <= 4
            group by t2.Hit
    ) as 'rol_ave'
from batting_ave t1
order by batter, game_date
limit 20






