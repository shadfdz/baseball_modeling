use baseball;

set @@max_heap_table_size=2342177280;
set SESSION sql_mode = 'ERROR_FOR_DIVISION_BY_ZERO ';

-- Create a rolling look up table for starting pitch
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_starting_pitch;
CREATE TEMPORARY TABLE t_rolling_lookup_starting_pitch ENGINE=MEMORY AS
Select g.game_id,
       g.gid,
       g.local_date,
       pc.pitcher,
       pc.Walk,
       pc.Hit,
       pc.endingInning,
       pc.startingInning,
       pc.pitchesThrown,
       pc.Home_Run,
       pc.Hit_By_Pitch,
       pc.Strikeout
from pitcher_counts pc
join game g on pc.game_id = g.game_id
order by local_date, pitcher;
CREATE UNIQUE INDEX rolling_lookup_date_game_batter_id_idx ON t_rolling_lookup_starting_pitch (game_id, pitcher, local_date);
CREATE UNIQUE INDEX rolling_lookup_game_batter_id_idx ON t_rolling_lookup_starting_pitch (game_id, pitcher);
CREATE UNIQUE INDEX rolling_lookup_date_batter_id_idx ON t_rolling_lookup_starting_pitch (local_date, pitcher);
CREATE INDEX rolling_lookup_game_id_idx ON t_rolling_lookup_starting_pitch (game_id);
CREATE INDEX rolling_lookup_local_date_idx ON t_rolling_lookup_starting_pitch (local_date);
CREATE INDEX rolling_lookup_batter_idx ON t_rolling_lookup_starting_pitch (pitcher);

-- Duplicate the rolling lookup for the table join
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_starting_pitch2;
CREATE TEMPORARY TABLE t_rolling_lookup_starting_pitch2
SELECT * FROM t_rolling_lookup_starting_pitch;

-- Create a temp table of rolling stats of starting pitch
DROP TEMPORARY TABLE IF EXISTS rolling_starting_pitch;
CREATE TEMPORARY TABLE rolling_starting_pitch ENGINE=MEMORY AS
Select rlsp.game_id,
       rlsp.local_date,
       rlsp.pitcher,
       SUM(rlsp2.Walk + rlsp2.Hit) / SUM(rlsp2.endingInning-rlsp2.startingInning + 1) as 'SP_WHIP',
       SUM(rlsp2.Strikeout)/SUM(rlsp2.Walk) as 'SP_KBB',
       SUM(((13 * rlsp2.Home_Run) + (3 * (rlsp2.Hit_By_Pitch+rlsp2.Walk)) - (2 * rlsp2.Strikeout))) / SUM((rlsp2.endingInning-rlsp2.startingInning + 1)) AS 'FIP'
from t_rolling_lookup_starting_pitch rlsp
JOIN t_rolling_lookup_starting_pitch2 rlsp2 on rlsp.pitcher = rlsp2.pitcher
AND rlsp2.local_date between date_sub(rlsp.local_date, interval 100 day ) and rlsp.local_date
GROUP BY rlsp.game_id, rlsp.local_date, rlsp.pitcher
order by local_date, rlsp.pitcher;
CREATE UNIQUE INDEX rolling_starting_pitch_date_game_pitch_idx ON rolling_starting_pitch (game_id, pitcher, local_date);
CREATE UNIQUE INDEX rolling_starting_pitch_game_pitch_idx ON rolling_starting_pitch (game_id, pitcher);
CREATE INDEX rolling_starting_pitch_game_idx ON rolling_starting_pitch (game_id);

DROP TEMPORARY TABLE IF EXISTS rolling_starting_pitch2;
CREATE TEMPORARY TABLE rolling_starting_pitch2 AS
SELECT * FROM rolling_starting_pitch;
CREATE UNIQUE INDEX rolling_starting_pitch_date_game_pitch_idx ON rolling_starting_pitch2 (game_id, pitcher, local_date);
CREATE UNIQUE INDEX rolling_starting_pitch_game_pitch_idx ON rolling_starting_pitch2 (game_id, pitcher);
CREATE INDEX rolling_starting_pitch_game_idx ON rolling_starting_pitch2 (game_id);


-- Create a rolling lookup of team pitching stats
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_team_pitching;
CREATE TEMPORARY TABLE t_rolling_lookup_team_pitching ENGINE=MEMORY AS
SELECT
g.game_id,
team_id,
local_date,
home_team_id,
hit,
Hit_By_Pitch,
walk,
atbat,
strikeout,
force_out,
ground_out,
Fly_Out,
Field_Error
from team_pitching_counts tpc
join game g on tpc.game_id = g.game_id
order by local_date, home_team_id;
CREATE UNIQUE INDEX rolling_lookup_date_game_team_id_idx ON t_rolling_lookup_team_pitching (game_id, team_id, local_date);
CREATE UNIQUE INDEX rolling_lookup_game_team_id_idx ON t_rolling_lookup_team_pitching (game_id, team_id);
CREATE INDEX rolling_lookup_team_id_idx ON t_rolling_lookup_team_pitching (game_id);
CREATE INDEX rolling_lookup_team_date_idx ON t_rolling_lookup_team_pitching (local_date);
CREATE INDEX rolling_lookup_team_idx ON t_rolling_lookup_team_pitching (team_id);

-- Duplicate team rolling lookup for table join
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_team_pitching2;
CREATE TEMPORARY TABLE t_rolling_lookup_team_pitching2
Select * from t_rolling_lookup_team_pitching;

-- Create table of rolling team pitch stats
DROP TEMPORARY TABLE IF EXISTS rolling_team_pitch_stats;
CREATE TEMPORARY TABLE rolling_team_pitch_stats AS
SELECT
rlp1.team_id,
rlp1.game_id,
rlp1.local_date,
SUM(rlp1.Hit + rlp1.Hit_By_Pitch + rlp1.Walk) / SUM(rlp1.atBat + rlp1.Walk + rlp1.Hit_By_Pitch) AS 'HitWalks',
SUM(rlp1.atBat) / SUM(rlp1.Strikeout) AS 'atBatStrikeout',
SUM(rlp1.Force_Out + rlp1.Ground_Out + rlp1.Fly_Out) / SUM(rlp1.atbat) AS 'HitOuts',
SUM(rlp1.Field_Error) / SUM(rlp1.walk + rlp1.hit) AS 'atBatFieldError',
SUM(rlp1.Strikeout) / SUM(rlp1.Walk + rlp1.hit) AS 'StrikeWalk'
FROM t_rolling_lookup_team_pitching rlp1
JOIN t_rolling_lookup_team_pitching2 rlp2 on rlp1.team_id = rlp2.team_id
AND rlp2.local_date between date_sub(rlp1.local_date, interval 100 day ) and rlp1.local_date
GROUP BY rlp1.team_id, rlp1.game_id, rlp1.local_date
order by rlp1.local_date, rlp1.team_id;
CREATE UNIQUE INDEX rolling_team_pitch_game_team_idx ON rolling_team_pitch_stats (game_id, team_id);
CREATE INDEX rolling_team_pitch_game_idx ON rolling_team_pitch_stats (game_id);
CREATE INDEX rolling_team_pitch_team_idx ON rolling_team_pitch_stats (team_id);



-- Duplicate rolling team pitch stats
DROP TEMPORARY TABLE IF EXISTS rolling_team_pitch_stats2;
CREATE TEMPORARY TABLE rolling_team_pitch_stats2 AS
SELECT * from rolling_team_pitch_stats;
CREATE UNIQUE INDEX rolling_team_pitch_game_team_idx ON rolling_team_pitch_stats2 (game_id, team_id);
CREATE INDEX rolling_team_pitch_game_idx ON rolling_team_pitch_stats2 (game_id);
CREATE INDEX rolling_team_pitch_team_idx ON rolling_team_pitch_stats2 (team_id);



-- Create temp look up table for rolling stats for team batting
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_team_batting;
CREATE TEMPORARY TABLE t_rolling_lookup_team_batting ENGINE=MEMORY AS
SELECT
		g.game_id,
		local_date,
		team_id,
		atBat,
		Hit,
       stolenBaseHome,
       stolenBase2B,
       stolenBase3B,
        caughtStealing2B,
       caughtStealing3B,
       caughtStealingHome,
       Double_Play,
       Runner_Out,
       Home_Run,
       Sac_Bunt,
       Sac_Fly,
       Sac_Fly_DP,
       Walk,
       Strikeout
    FROM team_batting_counts bc
    JOIN game g ON g.game_id = bc.game_id
	ORDER BY team_id, local_date;
CREATE UNIQUE INDEX rolling_lookup_date_game_batter_id_idx ON t_rolling_lookup_team_batting (game_id, team_id, local_date);
CREATE UNIQUE INDEX rolling_lookup_game_batter_id_idx ON t_rolling_lookup_team_batting (game_id, team_id);
CREATE INDEX rolling_lookup_game_id_idx ON t_rolling_lookup_team_batting (game_id);
CREATE INDEX rolling_lookup_local_date_idx ON t_rolling_lookup_team_batting (local_date);
CREATE INDEX rolling_lookup_batter_idx ON t_rolling_lookup_team_batting (team_id);

-- Duplicate temp table rolling stats for team batting
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_team_batting2;
CREATE TEMPORARY TABLE t_rolling_lookup_team_batting2
Select * from t_rolling_lookup_team_batting;

-- Create table for rolling 100 days stats for team batting
DROP TEMPORARY TABLE IF EXISTS rolling_team_batting_stats;
CREATE TEMPORARY TABLE rolling_team_batting_stats AS
SELECT
    rlb1.local_date,
    rlb1.game_id,
    rlb1.team_id,
    SUM(rlb1.stolenBaseHome + rlb1.stolenBase2B + rlb1.stolenBase3B) / SUM(rlb1.Hit + rlb1.Walk - rlb1.Home_Run) AS 'StolenBaseBB',
       SUM(rlb1.Home_Run)/SUM(rlb1.Hit) AS 'HRHIT',
       SUM(rlb1.Hit)/sum(rlb1.atBat) AS 'BA',
       sum(rlb1.Strikeout)/sum(rlb1.atBat) AS 'StrikeAtBat',
       sum(rlb1.Sac_Fly + rlb1.Sac_Bunt + rlb1.Sac_Fly_DP) / SUM(rlb1.Hit) AS 'OUTSAC'
FROM t_rolling_lookup_team_batting rlb1
JOIN t_rolling_lookup_team_batting2 rlb2 on rlb1.team_id = rlb2.team_id
AND rlb2.local_date between date_sub(rlb1.local_date, interval 100 day ) and rlb1.local_date
GROUP BY rlb1.local_date, rlb1.game_id, rlb1.team_id
ORDER BY rlb1.local_date, rlb1.team_id;
CREATE UNIQUE INDEX rolling_team_batting_game_team_idx ON rolling_team_batting_stats (game_id, team_id);
CREATE INDEX rolling_team_batting_game_idx ON rolling_team_batting_stats (game_id);
CREATE INDEX rolling_team_batting_team_idx ON rolling_team_batting_stats (team_id);


-- Create copy of table for rolling 100 days stats for team batting
DROP TEMPORARY TABLE IF EXISTS rolling_team_batting_stats2;
CREATE TEMPORARY TABLE rolling_team_batting_stats2 AS
SELECT * from rolling_team_batting_stats;
CREATE UNIQUE INDEX rolling_team_batting_game_team_idx ON rolling_team_batting_stats2 (game_id, team_id);
CREATE INDEX rolling_team_batting_game_idx ON rolling_team_batting_stats2 (game_id);
CREATE INDEX rolling_team_batting_team_idx ON rolling_team_batting_stats2 (team_id);







DROP TABLE IF EXISTS baseball_stats;
CREATE TABLE baseball_stats AS
select
tr.series_streak,
tr.home_streak,
tr.win_lose,
tr.local_date,
rsp.FIP,
rsp.SP_WHIP,
rsp.SP_KBB,
rsp_away.O_SP_KBB,
rsp_away.O_SP_WHIP,
rsp_away.O_FIP,
rtbs.BA,
rtbs.HRHIT,
rtbs.StolenBaseBB,
rtbs.StrikeAtBat as 'TeamStrikeAtBat',
rtbs.OUTSAC,
rtbs2.BA AS 'O_BA',
rtbs2.HRHIT AS 'O_HRIHT',
rtbs2.StolenBaseBB AS 'O_StolenBaseBB',
rtbs2.StrikeAtBat AS 'StrikeAtBat',
rtbs2.OUTSAC AS 'O_OUTSAC',
rtps.atBatFieldError AS 'TeamatBatFieldError',
rtps.HitOuts,
rtps.StrikeWalk,
rtps.atBatStrikeout,
rtps2.atBatFieldError AS 'O_atBatFieldError',
rtps2.HitOuts AS 'O_HitOuts',
rtps2.StrikeWalk AS 'O_StrikeWalk',
rtps2.atBatStrikeout AS 'O_atBatStrikeout'
from team_results tr
join pitcher_counts pc on tr.game_id = pc.game_id
join rolling_starting_pitch rsp on tr.game_id = rsp.game_id and pc.pitcher = rsp.pitcher
join (
        select
            pc.game_id,
            pc.pitcher as 'O_Pitcher',
            rsp2.FIP as 'O_FIP',
            rsp2.SP_KBB as 'O_SP_KBB',
            rsp2.SP_WHIP as 'O_SP_WHIP'
        from pitcher_counts pc
        join rolling_starting_pitch2 rsp2 on pc.game_id = rsp2.game_id and pc.pitcher = rsp2.pitcher
        where startingPitcher = 1 and pc.awayTeam = 1
    ) as rsp_away on tr.game_id = rsp_away.game_id and pc.pitcher = rsp_away.O_Pitcher
join rolling_team_batting_stats rtbs on tr.game_id = rtbs.game_id and tr.team_id = rtbs.team_id
join rolling_team_batting_stats2 rtbs2 on tr.game_id = rtbs2.game_id and tr.opponent_id = rtbs2.team_id
join rolling_team_pitch_stats rtps on tr.game_id = rtps.game_id and tr.team_id = rtps.team_id
join rolling_team_pitch_stats2 rtps2 on tr.game_id = rtps2.game_id and tr.opponent_id = rtps2.team_id
WHERE home_away = 'H'
order by tr.local_date desc;