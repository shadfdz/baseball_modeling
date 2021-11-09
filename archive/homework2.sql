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
    -- fill null values to avoid zero division and avg batting average
    IFNULL(ROUND(AVG(bc.Hit /NULLIF(bc.atBat,0)),3),0) AS 'batting_avg'
FROM baseball.game bg
LEFT JOIN batter_counts bc on bg.game_id = bc.game_id
-- remove batters without ids
WHERE BATTER is not null
GROUP BY bc.batter, year
ORDER BY batter, year asc;

-- Create a table the annual batting average of each player
DROP TABLE IF EXISTS f_annual_batting_avg;
CREATE TABLE f_annual_batting_avg
SELECT
	YEAR(bg.local_date) AS 'year',
    batter,
    -- fill null values to avoid zero division and avg batting average
    IFNULL(ROUND(AVG(bc.Hit /NULLIF(bc.atBat,0)),3),0) AS 'batting_avg'
FROM baseball.game bg
LEFT JOIN batter_counts bc on bg.game_id = bc.game_id
-- remove batters without ids
WHERE BATTER is not null
GROUP BY bc.batter, year
ORDER BY batter, year asc;


-- Question 2
-- Select historical batting average for each player
SELECT 
	bc.batter,
    -- fill null values to avoid zero division and get avg
    IFNULL(ROUND(AVG(bc.Hit /NULLIF(bc.atBat,0)),3),0) AS 'batting_avg'
FROM baseball.game bg
LEFT JOIN batter_counts bc on bg.game_id = bc.game_id
-- remove batters without ids
WHERE BATTER IS NOT NULL
GROUP BY bc.batter;

-- Create a table of historical batting average for each player
DROP TABLE IF EXISTS f_historical_batting_avg;
CREATE TABLE f_historical_batting_avg
SELECT
	bc.batter,
    -- fill null values to avoid zero division and get avg
    IFNULL(ROUND(AVG(bc.Hit /NULLIF(bc.atBat,0)),3),0) AS 'batting_avg'
FROM baseball.game bg
LEFT JOIN batter_counts bc on bg.game_id = bc.game_id
-- remove batters without ids
WHERE BATTER IS NOT NULL
GROUP BY bc.batter;

-- Question Number 3
-- The following queries are temporary tables that will be used to create
-- a final table the will show the 100 day rolling average of each batter's
-- batting average. The rolling average includes non playing days and will
-- count as zero.

-- 3.1 Create temp table for average batting ave of each player per day
DROP TEMPORARY TABLE if exists t_batting_ave;
CREATE TEMPORARY TABLE t_batting_ave
SELECT
    -- convert to date from datetime
    date(bg.local_date) as 'game_date',
    batter,
    -- avg batting average by day since players can play multiple games per day
    avg(Hit/nullif(atBat,0)) as batt_ave
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
GROUP BY date(bg.local_date), batter;

-- 3.2 Create a similar copy of t_batting_average for upcoming table join
DROP TEMPORARY TABLE if exists t_batting_ave_2;
CREATE TEMPORARY TABLE t_batting_ave_2
SELECT *
FROM t_batting_ave;

-- 3.3 Create a table of all dates between the oldest and the newest date in dataset
DROP TEMPORARY TABLE if exists DateRange;
-- increase recursive depth for data creation
SET SESSION cte_max_recursion_depth = 1000000;
CREATE TEMPORARY TABLE DateRange
-- recursively create a range of dates from min game_date to max game date
-- in DATE format yyyy-mm-dd
WITH RECURSIVE DateRange (DateName) AS
(
  SELECT (select date(min(game_date)) from t_batting_ave)
    UNION ALL
  SELECT adddate(DateName, 1) FROM DateRange WHERE DateName < (select date(max(game_date)) from t_batting_ave_2)
)
SELECT
       DateName
FROM DateRange;

-- 3.4 Create range of dates for each batter (dates b/w oldest and newest date played)
-- using Cross Join with DateRange table. Coalesce null values for batting avg (for non playing days) to zero.
-- Source: https://stackoverflow.com/questions/19075098/how-to-fill-missing-dates-by-groups-in-a-table-in-sql?rq=1
DROP TEMPORARY TABLE if exists t_baseball_batting_ave;
CREATE TEMPORARY TABLE t_baseball_batting_ave
    SELECT
        batting_date.DateName,
        batting_date.batter,
        -- Coalesce null values for batting avg (for non playing days) to zero
        round(coalesce(ba2.batt_ave, 0), 3) battingave,
        ba2.batt_ave batting_ave
    FROM
        (
        -- Cross join DateRange table to inner query to get a date range
        -- between oldest and earliest date for each player
        SELECT
               batter, DateName
        FROM
            (
            -- select the oldest and newest date each batter has played
            SELECT
                   batter,
                   min(game_date) as lowest_date,
                   max(game_date) as highest_date
            FROM t_batting_ave
            GROUP BY batter
            ) date_param CROSS JOIN DateRange date_ranges
            WHERE date_ranges.DateName BETWEEN date_param.lowest_date and date_param.highest_date
        ) batting_date
            LEFT JOIN t_batting_ave_2 ba2 ON batting_date.batter = ba2.batter
                                                 AND batting_date.DateName = ba2.game_date
    ORDER BY batter;

-- 3.5 Create a temporary table and calculate 100 day rolling average
DROP TEMPORARY TABLE if exists t_baseball_rolling_ave;
CREATE TEMPORARY TABLE t_baseball_rolling_ave
    SELECT
        DateName AS game_date,
        batter,
        battingave AS batting_avg,
        -- get 100 day rolling average of batting average of each batter
        round(avg(battingave) OVER (PARTITION BY batter ORDER BY DateName ASC ROWS 100 PRECEDING),3) 100_day_rollavg
    FROM t_baseball_batting_ave
    ORDER BY batter, DateName;

-- 3.6 Create table of 100 day rollin average for each batter
-- remove non playing days where batting_avg was null during coalesce
DROP TABLE IF EXISTS f_baseball_100_day_bat_avg;
CREATE TABLE f_baseball_100_day_bat_avg
SELECT
    game_date,
    batter,
    batting_avg,
    100_day_rollavg
FROM t_baseball_rolling_ave
WHERE batting_avg IS NOT NULL;

-- To Test the Data, enter a batter id and a valid game date to
-- retrieve the 100 day rolling average by means of the rolling average
-- column and by summing batting average and dividing the sum by 100.
-- If both values are equal then the 100 day rolling average column is legit yo

SET @batter_id = '110029';
SET @rolling_ave_date = '2007-06-29';
SELECT
    CASE WHEN
    (SELECT round(sum(batting_avg)/100,3)
    FROM f_baseball_100_day_bat_avg
    WHERE game_date < @rolling_ave_date
    AND batter = @batter_id)
    =
    (SELECT `100_day_rollavg`
    FROM f_baseball_100_day_bat_avg
    WHERE game_date = @rolling_ave_date
    AND batter = @batter_id)
    THEN ('Equal.')
ELSE 'Unequal'
END;

-- BONUS for reading through all of that. Sit back and select dancing dude
drop table if exists t_Dance;
create temporary table t_Dance (Reaction varchar(20));
insert into t_Dance (Reaction)
values
       ('L(*_*)'),
       ('  ))z'),
       (' / l');

Select *
from t_Dance;