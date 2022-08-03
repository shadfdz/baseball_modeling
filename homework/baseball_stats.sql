use baseball;
set @@max_heap_table_size=2342177280;

/*****************************************************************************
                            Starting Pitch Stats
 *****************************************************************************/
-- CREATE ROLLING STATS FOR PITCHER
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_starting_pitch;
CREATE TEMPORARY TABLE t_rolling_lookup_starting_pitch ENGINE=MEMORY AS
Select
       local_date,
       g.game_id,
       pitcher,
       SUM(Walk) OVER (PARTITION BY pitcher ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_walk,
       SUM(plateApperance) OVER (PARTITION BY pitcher ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_pa,
       SUM(toBase) OVER (PARTITION BY pitcher ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_tbase,
       SUM(Single + pc.Double + Triple) OVER (PARTITION BY pitcher ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_sdt,
       SUM(Hit) OVER (PARTITION BY pitcher ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_hit,
       SUM(endingInning - startingInning + 1) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_innings,
       AVG((endingInning - startingInning) + 1) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_a_innings,
       AVG(DaysSinceLastPitch) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_a_daysince,
       SUM(pitchesThrown) OVER (PARTITION BY pitcher ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_pitch_thrown,
       SUM(Home_Run) OVER (PARTITION BY pitcher ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_home_run,
       SUM(Hit_By_Pitch) OVER (PARTITION BY pitcher ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_hit_by_pitch,
       SUM(Strikeout) OVER (PARTITION BY pitcher ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_strikeout,
       AVG(CAST(Strikeout as float )) OVER (PARTITION BY pitcher ORDER BY local_date asc ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING) as r_strikeout_50
from pitcher_counts pc
join game g on pc.game_id = g.game_id;


DROP TEMPORARY TABLE IF EXISTS rolling_starting_pitch;
CREATE TEMPORARY TABLE rolling_starting_pitch ENGINE=MEMORY AS
Select rlsp.game_id,
       rlsp.local_date,
       rlsp.pitcher,
       CASE
           WHEN rlsp.r_innings = 0 then 0.0000
           else (rlsp.r_walk + rlsp.r_hit) / rlsp.r_innings
        END as 'SP_WHIP',
       rlsp.r_strikeout/rlsp.r_walk as 'SP_KBB',
       CASE
           WHEN rlsp.r_innings = 0 then 0.0000
           ELSE ((13 * rlsp.r_home_run) + (3 * (rlsp.r_hit_by_pitch + rlsp.r_walk)) - (2 * rlsp.r_strikeout)) / (rlsp.r_innings)
        END as 'SP_FIP',
        CASE
           WHEN rlsp.r_pitch_thrown= 0 then 0.0000
           else (rlsp.r_sdt * rlsp.r_pa/rlsp.r_pitch_thrown)
        END as 'SP_sdt_pt',
       r_a_innings,
        CASE
           WHEN rlsp.r_pitch_thrown = 0 then 0.0000
           else (rlsp.r_hit /rlsp.r_pitch_thrown)
        END as 'SP_hit_pt',
        CASE
           WHEN rlsp.r_pitch_thrown = 0 then 0.0000
           else (rlsp.r_tbase /rlsp.r_pitch_thrown)
        END as 'SP_tb_pt',
       r_a_daysince,
       r_strikeout_50
from t_rolling_lookup_starting_pitch rlsp;
CREATE INDEX rolling_lookup_game_id_idx ON rolling_starting_pitch (game_id);
CREATE INDEX rolling_lookup_batter_idx ON rolling_starting_pitch (pitcher);

-- Duplicate Table
DROP TEMPORARY TABLE IF EXISTS rolling_starting_pitch2;
CREATE TEMPORARY TABLE rolling_starting_pitch2 ENGINE=MEMORY AS
SELECT * FROM rolling_starting_pitch;
CREATE INDEX rolling_lookup_game_id_idx ON rolling_starting_pitch2 (game_id);
CREATE INDEX rolling_lookup_batter_idx ON rolling_starting_pitch2 (pitcher);

-- drop temp table
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_starting_pitch;
/*****************************************************************************
                        Team Pitching Rolling Stats by Team
 *****************************************************************************/
-- CREATE ROLLING STATS FOR TEAM PITCHING
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_team_pitching;
CREATE TEMPORARY TABLE t_rolling_lookup_team_pitching ENGINE=MEMORY AS
SELECT
g.game_id,
team_id,
local_date,
       SUM(Hit) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_t_hit,
       SUM(Hit_By_Pitch) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_t_hit_pitch,
       SUM(walk) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_t_walk,
       SUM(atBat) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_t_atBat,
       SUM(Strikeout) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_t_so,
       SUM(Force_Out + Ground_Out + Fly_Out) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_t_forceout,
       AVG(CAST(Field_Error AS FLOAT )) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_t_a_fielderror,
       AVG(Strikeout) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_t_a_so_10,
       AVG(Strikeout + `Strikeout_-_TP` + `Strikeout_-_DP`) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_t_a_so,
       AVG(finalScore) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_t_a_score_10,
       SUM(Single + tpc.Double + Triple) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_t_sdt,
       AVG(Home_Run) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_t_a_hr,
       SUM(CAST(win as FLOAT )) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING) as r_t_a_win_50
from team_pitching_counts tpc
join game g on tpc.game_id = g.game_id;

DROP TEMPORARY TABLE IF EXISTS rolling_team_pitch_stats;
CREATE TEMPORARY TABLE rolling_team_pitch_stats AS
SELECT
rlp.game_id,
rlp.local_date,
rlp.team_id,
CASE
    WHEN ((rlp.r_t_atBat + rlp.r_t_walk + rlp.r_t_hit_pitch)  = 0 ) then 0.0
    else (rlp.r_t_hit + rlp.r_t_hit_pitch + rlp.r_t_walk) / (rlp.r_t_atBat + rlp.r_t_walk + rlp.r_t_hit_pitch)
END AS 'HitWalks',
CASE
    WHEN (rlp.r_t_so  = 0 ) then 0.0000
    else (rlp.r_t_atBat / rlp.r_t_so)
END AS 'atBatStrikeout',
CASE
    WHEN (rlp.r_t_atBat  = 0 ) then 0.0000
    else (rlp.r_t_forceout / rlp.r_t_atBat)
END  AS 'HitOuts',
rlp.r_t_a_so_10,
CASE
    WHEN (rlp.r_t_a_so = 0 ) then 0.0000
    else (rlp.r_t_sdt / rlp.r_t_a_so)
END  AS 'sdt_force_outs',
CASE
    WHEN (rlp.r_t_forceout = 0 ) then 0.0000
    else (rlp.r_t_a_score_10 / rlp.r_t_forceout)
END  AS 'score_fo',
r_t_a_fielderror,
CASE
    WHEN (rlp.r_t_a_so_10 = 0 ) then 0.0000
    else (rlp.r_t_a_hr / rlp.r_t_a_so_10)
END  AS 'hr_so',
rlp.r_t_a_win_50
FROM t_rolling_lookup_team_pitching rlp;
CREATE INDEX rolling_lookup_game_id_idx ON rolling_team_pitch_stats (game_id);
CREATE INDEX rolling_lookup_team_id_idx ON rolling_team_pitch_stats (team_id);


DROP TEMPORARY TABLE IF EXISTS rolling_team_pitch_stats2;
CREATE TEMPORARY TABLE rolling_team_pitch_stats2 AS
SELECT * FROM rolling_team_pitch_stats;
CREATE INDEX rolling_lookup_game_id_idx ON rolling_team_pitch_stats2 (game_id);
CREATE INDEX rolling_lookup_team_id_idx ON rolling_team_pitch_stats2 (team_id);

# drop unused temp tables
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_team_pitching;

/*****************************************************************************
                        Team Batting Rolling Stats by Team
 *****************************************************************************/
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_team_batting;
CREATE TEMPORARY TABLE t_rolling_lookup_team_batting ENGINE=MEMORY AS
SELECT
       g.game_id,
       local_date,
       team_id,
       SUM(atBat) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_p_atBat,
       SUM(Hit) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_p_hit,
       SUM(stolenBaseHome + stolenBase2B + stolenBase3B) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_p_stolen,
       SUM(caughtStealingHome + bc.caughtStealing2B + bc.caughtStealing3B) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_p_caught,
       SUM(Double_Play + Triple_Play) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_p_dt_play,
       SUM(Home_Run) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_p_homerun,
       SUM(Sac_Bunt + Sac_Fly + Sac_Fly_DP) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_p_sac,
       SUM(Walk) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_p_walk,
       SUM(Strikeout + `Strikeout_-_DP` + `Strikeout_-_TP`) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_p_strikeout,
       AVG(bc.finalScore) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_p_a_score,
       AVG(bc.plateApperance) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 25 PRECEDING AND 1 PRECEDING) as r_p_pa,
       SUM(CAST(win as FLOAT)) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING) as r_p_win_50
    FROM team_batting_counts bc
    JOIN game g ON g.game_id = bc.game_id;

DROP TEMPORARY TABLE IF EXISTS rolling_team_batting_stats;
CREATE TEMPORARY TABLE rolling_team_batting_stats AS
SELECT
rlb.team_id,
rlb.game_id,
rlb.local_date,
rlb.r_p_homerun / 10 as 'ave_rolling_hr',
CASE
    WHEN ((rlb.r_p_hit + rlb.r_p_walk - rlb.r_p_homerun) = 0) then 0.0
    else rlb.r_p_stolen / (rlb.r_p_hit + rlb.r_p_walk - rlb.r_p_homerun)
END AS 'StolenBperNonHR',
CASE
    WHEN (rlb.r_p_hit = 0) then 0.0000
    else rlb.r_p_homerun / rlb.r_p_hit
END  as 'hrHits',
CASE
    WHEN (rlb.r_p_atBat = 0) then 0.0000
    else rlb.r_p_hit / rlb.r_p_atBat
END as 'BA',
CASE
    WHEN (rlb.r_p_atBat = 0) then 0.0000
    else rlb.r_p_strikeout / rlb.r_p_atBat
END  as 'StrikAtBat',
CASE
    WHEN (rlb.r_p_hit = 0) then 0.0000
    else rlb.r_p_sac / rlb.r_p_hit
END  as 'SacHits',
CASE
    WHEN (rlb.r_p_pa = 0) then 0.0000
    else rlb.r_p_a_score / rlb.r_p_pa
END  as 'r_p_score_pa',
r_p_win_50
FROM t_rolling_lookup_team_batting rlb;
CREATE INDEX rolling_lookup_game_id_idx ON rolling_team_batting_stats (game_id);
CREATE INDEX rolling_lookup_team_id_idx ON rolling_team_batting_stats (team_id);

DROP TEMPORARY TABLE IF EXISTS rolling_team_batting_stats2;
CREATE TEMPORARY TABLE rolling_team_batting_stats2 AS
SELECT * FROM rolling_team_batting_stats;
CREATE INDEX rolling_lookup_game_id_idx ON rolling_team_batting_stats2 (game_id);
CREATE INDEX rolling_lookup_team_id_idx ON rolling_team_batting_stats2 (team_id);

-- Drop unused lookup tables
DROP TEMPORARY TABLE IF EXISTS t_rolling_lookup_team_batting;

/*****************************************************************************
                        Create tables for avg pitchers in game
 *****************************************************************************/
DROP TEMPORARY TABLE IF EXISTS pitcher_d_count;
CREATE TEMPORARY TABLE pitcher_d_count AS
select local_date, g.game_id, team_id, count(distinct pitcher) as dpitchers
from pitcher_counts
join game g on pitcher_counts.game_id = g.game_id
group by game_id, team_id;

DROP TEMPORARY TABLE IF EXISTS pitcher_d_rcount;
CREATE TEMPORARY TABLE pitcher_d_rcount AS
select
       local_date
       game_id,
       team_id,
       AVG(dpitchers) OVER (PARTITION BY team_id ORDER BY local_date asc ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) as r_a_dpitchers
from pitcher_d_count;

DROP TEMPORARY TABLE IF EXISTS pitcher_d_rcount2;
CREATE TEMPORARY TABLE pitcher_d_rcount2 AS
select *
from pitcher_d_rcount;

-- drop unused lookup table
DROP TEMPORARY TABLE IF EXISTS pitcher_d_count;

/*****************************************************************************
                        Duplicate Tables for Join
 *****************************************************************************/
DROP TEMPORARY TABLE IF EXISTS team_results2;
CREATE TEMPORARY table team_results2 as
SELECT *
FROM team_results;
CREATE INDEX rolling_lookup_game_id_idx ON team_results2 (game_id);
CREATE INDEX rolling_lookup_team_id_idx ON team_results2 (team_id);
CREATE INDEX rolling_lookup_opponent_id_idx ON team_results2 (opponent_id);

DROP TEMPORARY TABLE IF EXISTS pitcher_counts2;
create temporary table pitcher_counts2 as
select *
from pitcher_counts;


/*****************************************************************************
                        Create Final Table
 *****************************************************************************/
DROP TABLE IF EXISTS baseball_stats;
CREATE TABLE baseball_stats ENGINE=MEMORY AS
select tr.local_date, YEAR(tr.local_date) as 'Year', tr.win_lose,
ifnull((rsp.SP_WHIP - rsp2.SP_WHIP), 0) as 'SP_WHIP_d',
ifnull((rsp.SP_KBB - rsp2.SP_KBB), 0) as 'SP_KBB_d',
ifnull((rsp.SP_FIP - rsp2.SP_FIP), 0) as 'SP_FIP_d',
ifnull((rsp.SP_sdt_pt - rsp2.SP_sdt_pt), 0) as 'SP_sdt_pt',
ifnull((rsp.r_a_innings - rsp2.r_a_innings), 0) as 'SP_a_innings',
ifnull((rsp.SP_hit_pt - rsp2.SP_hit_pt), 0) as 'SP_hit_pt',
ifnull((rsp.SP_tb_pt - rsp2.SP_tb_pt), 0) as 'SP_tb_pt',
ifnull((rsp.r_a_daysince - rsp2.r_a_daysince), 0) as 'SP_a_days_since',
ifnull((rsp.r_strikeout_50 - rsp2.r_strikeout_50), 0) as 'SP_so_50',
ifnull((rtps.HitWalks - rtps2.HitWalks), 0) as 'TP_HitWalks_d',
ifnull((rtps.atBatStrikeout - rtps2.atBatStrikeout), 0) as 'TP_atBatStrike',
ifnull((rtps.HitOuts - rtps2.HitOuts), 0) as 'TP_HitOuts',
ifnull((rtps.r_t_a_so_10- rtps2.r_t_a_so_10), 0) as 'TP_a_so_10',
ifnull((rtps.sdt_force_outs - rtps2.sdt_force_outs), 0) as 'TP_sdt_fo',
ifnull((rtps.score_fo - rtps2.score_fo), 0) as 'TP_score_fo',
ifnull((rtps.r_t_a_fielderror - rtps2.r_t_a_fielderror), 0) as 'TP_a_fe',
ifnull((rtps.hr_so - rtps2.hr_so), 0) as 'TP_hr_so',
ifnull((rtbs.ave_rolling_hr - rtbs2.ave_rolling_hr), 0) as 'TB_ARHR',
ifnull((rtbs.hrHits - rtbs2.hrHits), 0) as 'TB_HrHits',
ifnull((rtbs.BA - rtbs2.BA), 0) as 'TB_BA',
ifnull((rtbs.StrikAtBat - rtbs2.StrikAtBat), 0) as 'TB_strike_AB_d',
ifnull((rtbs.SacHits - rtbs2.SacHits), 0) as 'TB_SacHits_d',
ifnull((rtbs.r_p_score_pa- rtbs2.r_p_score_pa), 0) as 'TB_score_pa',
ifnull((rtbs.r_p_win_50 - rtbs2.r_p_win_50), 0) as 'T_win_50'
from team_results tr
join (
    SELECT pc.game_id, pc.team_id, pc.pitcher
    from pitcher_counts pc
    where pc.game_id and pc.homeTeam = 1 and pc.startingPitcher = 1
) home_pitch on tr.game_id = home_pitch.game_id
join (
    SELECT pc2.game_id, pc2.team_id, pc2.pitcher
    from pitcher_counts2 pc2
    where game_id and pc2.awayTeam = 1 and pc2.startingPitcher = 1
) away_pitch on home_pitch.game_id = away_pitch.game_id
join rolling_starting_pitch rsp on tr.game_id = rsp.game_id and home_pitch.pitcher = rsp.pitcher
join rolling_starting_pitch2 rsp2 on tr.game_id = rsp2.game_id and away_pitch.pitcher = rsp2.pitcher
join rolling_team_batting_stats rtbs on tr.game_id = rtbs.game_id and tr.team_id = rtbs.team_id
join rolling_team_batting_stats2 rtbs2 on tr.game_id = rtbs2.game_id and tr.opponent_id = rtbs2.team_id
join rolling_team_pitch_stats rtps on tr.game_id = rtps.game_id and tr.team_id = rtps.team_id
join rolling_team_pitch_stats2 rtps2 on tr.game_id = rtps2.game_id and tr.opponent_id = rtps2.team_id
where home_away = 'H'
order by tr.local_date asc;

