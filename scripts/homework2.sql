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


-- Question Number 3

-- Create temp table for average batting ave of each player per day
DROP TEMPORARY TABLE if exists batting_ave;
CREATE TEMPORARY TABLE batting_ave
SELECT
    date(bg.local_date) as 'game_date',
    batter,
    avg(Hit/nullif(atBat,0)) as batt_ave
FROM game bg
JOIN batter_counts bc on bg.game_id = bc.game_id
GROUP BY date(bg.local_date), batter;


DROP TEMPORARY TABLE if exists batting_ave_2;
CREATE TEMPORARY TABLE batting_ave_2
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
  SELECT (select date(min(game_date)) from batting_ave)
    UNION ALL
  SELECT adddate(DateName, 1) FROM DateRange WHERE DateName < (select date(max(game_date)) from batting_ave_2)
)
SELECT
       DateName
FROM DateRange;

-- Cross join DateRange table to batting_ave
-- and coalesce missing values
-- Source: https://stackoverflow.com/questions/19075098/how-to-fill-missing-dates-by-groups-in-a-table-in-sql?rq=1
DROP TEMPORARY TABLE if exists baseball_batting_ave;
CREATE TEMPORARY TABLE baseball_batting_ave
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
            FROM batting_ave
            GROUP BY batter ) date_param CROSS JOIN DateRange date_ranges
            WHERE date_ranges.DateName BETWEEN date_param.lowest_date and date_param.highest_date
            ) batting_date LEFT JOIN batting_ave_2 ba2 ON batting_date.batter = ba2.batter AND batting_date.DateName = ba2.game_date
    ORDER BY batter;

-- Create a temporary table and calculate 100 day rolling average
DROP TEMPORARY TABLE if exists baseball_rolling_ave;
CREATE TEMPORARY TABLE baseball_rolling_ave
    SELECT
        DateName AS game_date,
        batter,
        battingave AS batting_avg,
        round(avg(battingave) OVER (PARTITION BY batter ORDER BY DateName ASC ROWS 100 PRECEDING),3) 100_day_rollavg
    FROM baseball_batting_ave
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
FROM baseball_rolling_ave
WHERE batting_avg IS NOT NULL;

