/*
Homework 2
Shad Fernandez
Date: 19-SEP-2021
*/

-- double check numbers, null values, and rounding rules
-- use baseball;

# 1
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

# 2
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

# 3

DROP TEMPORARY TABLE if exists batting_ave;
CREATE TEMPORARY TABLE batting_ave
Select
date(bg.local_date) as 'game_date',
count(distinct bc.game_id) as 'game_count',
batter,
sum(Hit) as 'Hits',
sum(atBat) as 'atBats'
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
group by date(bg.local_date), batter
order by game_count desc;

select *
from batting_ave
order by game_count desc
limit 200;


DROP TEMPORARY TABLE if exists batting_ave_2;
CREATE TEMPORARY TABLE batting_ave_2
Select
date(bg.local_date) as 'game_date',
count(distinct bc.game_id) as 'game_count',
batter,
sum(Hit) as 'Hits',
sum(atBat) as 'atBats'
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
group by date(bg.local_date), batter
order by game_count desc;

# -- select batter 407832, 112297

DROP TEMPORARY TABLE if exists DateRange;
-- set recursion depth to difference in max min date
-- change n later!
SET SESSION cte_max_recursion_depth = 1000000;
CREATE TEMPORARY TABLE DateRange
WITH RECURSIVE DateRange (DateName) AS
(
  SELECT (select date(min(game_date)) from batting_ave)
  UNION ALL
  SELECT adddate(DateName, 1) FROM DateRange WHERE DateName < (select date(max(game_date)) from batting_ave_2)
)
SELECT DateName
FROM DateRange;

DROP TEMPORARY TABLE if exists cleaned;
CREATE temporary table cleaned
select p.DateName, coalesce(a.Hits, 0) hits, p.batter, coalesce(a.game_count,1) game_count, coalesce(a.atBats, 0) atbats
from
(select batter, DateName
from
( select batter, min(game_date) as lowest_date, max(game_date) as highest_date
from batting_ave
group by batter ) q cross join DateRange b
where b.DateName between q.lowest_date and q.highest_date
) p left join batting_ave_2 a on p.batter = a.batter
    and p.DateName = a.game_date
order by batter;
