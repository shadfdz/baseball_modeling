/*
Homework 2
Shad Fernandez
Date: 19-SEP-2021
*/

-- double check numbers, null values, and rounding rules
-- use baseball;

-- Question 1
-- Select annual batting average of each player and create a table
SELECT
	YEAR(bg.local_date) AS 'year',
    batter,
    IFNULL(ROUND(AVG(bc.Hit /NULLIF(bc.atBat,0)),3),0) AS 'batting_avg'
FROM baseball.game bg
LEFT JOIN batter_counts bc on bg.game_id = bc.game_id
WHERE BATTER is not null
GROUP BY bc.batter, year
ORDER BY batter, year asc;

DROP TABLE IF EXISTS f_annual_batting_avg;
CREATE TABLE f_annual_batting_avg
SELECT
	YEAR(bg.local_date) AS 'year',
    batter,
    IFNULL(ROUND(AVG(bc.Hit /NULLIF(bc.atBat,0)),3),0) AS 'batting_avg'
FROM baseball.game bg
LEFT JOIN batter_counts bc on bg.game_id = bc.game_id
WHERE BATTER is not null
GROUP BY bc.batter, year
ORDER BY batter, year asc;


-- Question 2
-- Select historical batting average for each player
SELECT 
	bc.batter,
    IFNULL(ROUND(AVG(bc.Hit /NULLIF(bc.atBat,0)),3),0) AS 'batting_avg'
FROM baseball.game bg
LEFT JOIN batter_counts bc on bg.game_id = bc.game_id
WHERE BATTER IS NOT NULL
GROUP BY bc.batter;

DROP TABLE IF EXISTS f_historical_batting_avg;
CREATE TABLE f_historical_batting_avg
SELECT
	bc.batter,
    IFNULL(ROUND(AVG(bc.Hit /NULLIF(bc.atBat,0)),3),0) AS 'batting_avg'
FROM baseball.game bg
LEFT JOIN batter_counts bc on bg.game_id = bc.game_id
WHERE BATTER IS NOT NULL
GROUP BY bc.batter;

-- Question Number 3

-- Create temp table for average batting ave of each player per day
DROP TEMPORARY TABLE if exists t_batting_ave;
CREATE TEMPORARY TABLE t_batting_ave
SELECT
    date(bg.local_date) as 'game_date',
    batter,
    avg(Hit/nullif(atBat,0)) as batt_ave
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
GROUP BY date(bg.local_date), batter;


DROP TEMPORARY TABLE if exists t_batting_ave_2;
CREATE TEMPORARY TABLE t_batting_ave_2
SELECT
    date(bg.local_date) as 'game_date',
    batter,
    avg(Hit/nullif(atBat,0)) as batt_ave
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
GROUP BY date(bg.local_date), batter;

-- Create a table of all dates played for each batter
-- Max and Mix date played for each batter
DROP TEMPORARY TABLE if exists DateRange;
SET SESSION cte_max_recursion_depth = 1000000;
CREATE TEMPORARY TABLE DateRange
WITH RECURSIVE DateRange (DateName) AS
(
  SELECT (select date(min(game_date)) from t_batting_ave)
    UNION ALL
  SELECT adddate(DateName, 1) FROM DateRange WHERE DateName < (select date(max(game_date)) from t_batting_ave_2)
)
SELECT
       DateName
FROM DateRange;

-- Cross join DateRange table to batting_ave
-- and coalesce missing values
-- Source: https://stackoverflow.com/questions/19075098/how-to-fill-missing-dates-by-groups-in-a-table-in-sql?rq=1
DROP TEMPORARY TABLE if exists t_baseball_batting_ave;
CREATE TEMPORARY TABLE t_baseball_batting_ave
    SELECT
        batting_date.DateName,
        batting_date.batter,
        round(coalesce(ba2.batt_ave, 0), 3) battingave,
        ba2.batt_ave batting_ave
    FROM
        (
        SELECT
               batter, DateName
        FROM
            (
            SELECT
                   batter,
                   min(game_date) as lowest_date,
                   max(game_date) as highest_date
            FROM t_batting_ave
            GROUP BY batter ) date_param CROSS JOIN DateRange date_ranges
            WHERE date_ranges.DateName BETWEEN date_param.lowest_date and date_param.highest_date
            ) batting_date LEFT JOIN t_batting_ave_2 ba2 ON batting_date.batter = ba2.batter AND batting_date.DateName = ba2.game_date
    ORDER BY batter;

-- Create a temporary table and calculate 100 day rolling average
DROP TEMPORARY TABLE if exists t_baseball_rolling_ave;
CREATE TEMPORARY TABLE t_baseball_rolling_ave
    SELECT
        DateName AS game_date,
        batter,
        battingave AS batting_avg,
        round(avg(battingave) OVER (PARTITION BY batter ORDER BY DateName ASC ROWS 100 PRECEDING),3) 100_day_rollavg
    FROM t_baseball_batting_ave
    ORDER BY batter, DateName;

-- Create table of 100 day rollin average for each batter
-- remove non playing days
DROP TABLE IF EXISTS f_baseball_100_day_bat_avg;
CREATE TABLE f_baseball_100_day_bat_avg
SELECT
    game_date,
    batter,
    batting_avg,
    100_day_rollavg
FROM t_baseball_rolling_ave
WHERE batting_avg IS NOT NULL;

select *
from f_baseball_100_day_bat_avg
limit 20;