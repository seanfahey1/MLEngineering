drop table if exists batter_per_game;
drop table if exists batter_100;
drop table if exists pitcher_per_game;
drop table if exists pitcher_100;
drop table if exists game_features;
drop table if exists pregame_odds_best_final;

/*
create or replace table batter_per_game (primary key(batter_id, game)) as
select 
	bc.batter as batter_id
	, bc.team_id as teamid
	, g.`game_id`as game
	, bc.atBat as AB
 	, bc.Hit as hit 
 	, bc.Walk as walk
	, bc.Home_Run as HR
	, case when bc.atBat = 0 then 0 else bc.Hit / bc.atBat end as bavg
	, cast(g.local_date as date) as local_date
from batter_counts bc 
left join game g on g.game_id = bc.game_id
left join team t on t.team_id = bc.team_id
where substring_index(g.local_date, "-", 1) is not null
group by batter_id, local_date
order by batter_id asc, local_date desc
;

create or replace table batter_100 (primary key(batter_id, game)) as
select 
	p1.batter_id as batter_id
	, p1.teamid as teamid
	, p1.local_date
	, p1.game
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.batter_id = p2.batter_id then p2.AB end) as 100_day_ab
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.batter_id = p2.batter_id then p2.hit end) as 100_day_hit
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.batter_id = p2.batter_id then p2.walk end) as 100_day_walk
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.batter_id = p2.batter_id then p2.HR end) as 100_day_HR
	, avg(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.batter_id = p2.batter_id then p2.bavg end) as 100_day_bavg
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.batter_id = p2.batter_id then 1 end) as num_game
FROM batter_per_game p1
left join batter_per_game p2 on p1.batter_id = p2.batter_id 
group by p1.local_date, batter_id
order by p1.batter_id desc, p1.local_date asc
limit 10
;

select * from batter_100;
*/

create or replace table pitcher_per_game (primary key(pitcher, game_id)) as
select 
	pc.pitcher
	, pc.game_id
	, pc.team_id
	, pc.atBat
	, pc.plateApperance
	, pc.Hit
	, pc.Single
	, pc.Double
	, pc.Triple
	, pc.Home_Run
	, pc.Walk
	, case when pc.plateApperance = 0 then 0 else (pc.Walk + pc.Hit) / (pc.plateApperance / 3) end as WHIP
	, pc.Strikeout
	, pc.Hit_By_Pitch
	, cast(g.local_date as date) as local_date
from pitcher_counts pc
left join game g on g.game_id = pc.game_id
where substring_index(g.local_date, "-", 1) is not null
group by pitcher, local_date
order by pitcher asc, local_date desc
;

create or replace table pitcher_100 (primary key(pitcher, game_id)) as
select 
	p1.pitcher
	, p1.team_id
	, p1.local_date
	, p1.game_id
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then p2.atBat end) as atBat_100
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then p2.plateApperance end) as plateApperance_100
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then p2.Hit end) as Hit_100
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then p2.Single end) as Single_100
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then p2.Double end) as Double_100
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then p2.Triple end) as Triple_100
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then p2.Home_Run end) as Home_Run_100
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then p2.Walk end) as Walk_100
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then p2.WHIP end) as WHIP_100
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then p2.Strikeout end) as Strikeout_100
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then p2.Hit_By_Pitch end) as Hit_By_Pitch_100
	, sum(case when datediff(p1.local_date, p2.local_date) between 1 and 100 and p1.pitcher = p2.pitcher then 1 end) as num_game
FROM pitcher_per_game p1
left join pitcher_per_game p2 on p1.pitcher = p2.pitcher 
group by p1.local_date, pitcher
order by p1.pitcher desc, p1.local_date asc
;

select * from pitcher_100;

create or replace table pregame_odds_last_update as
select 
	po.*
from pregame_odds po
left join pregame_odds b on po.game_id = b.game_id and po.capture_date < b.capture_date
where b.capture_date is null
order by game_id asc
;

create or replace table pregame_odds_best_final (primary key(game_id)) as 
select
	game_id
	, max(home_line) as home_best_odds
	, max(away_line) as away_best_odds
from pregame_odds_last_update
group by game_id
;

select * from pregame_odds_best_final;

create or replace table game_features (primary key(game_id)) as
select
	g.game_id as game_id
	, case when tr.win_lose = "W" then 1 else 0 end as home_team_wins
	, g.home_team_id
	, g.away_team_id
	, g.local_date
  	, case when (g.home_w + g.home_l) > 0 then g.home_w / (g.home_w + g.home_l) else 0 end as home_win_rate
	, case when (g.away_w + g.away_l) > 0 then g.away_w / (g.away_w + g.away_l) else 0 end as away_win_rate
	, g.home_pitcher
	, g.away_pitcher
 	, hp.atBat_100 as home_pitcher_atBat_100
 	, ap.atBat_100 as away_pitcher_atBat_100
	, hp.plateApperance_100 as home_pitcher_plateApperance_100
	, ap.plateApperance_100 as away_pitcher_plateApperance_100
	, hp.Hit_100 as home_pitcher_Hit_100
	, ap.Hit_100 as away_pitcher_Hit_100
	, hp.Single_100 as home_pitcher_Single_100
	, ap.Single_100 as away_pitcher_Single_100
	, hp.Double_100 as home_pitcher_Double_100
	, ap.Double_100 as away_pitcher_Double_100
	, hp.Triple_100 as home_pitcher_Triple_100
	, ap.Triple_100 as away_pitcher_Triple_100
	, hp.Home_Run_100 as home_pitcher_Home_Run_100
	, ap.Home_Run_100 as away_pitcher_Home_Run_100
	, hp.Walk_100 as home_pitcher_Walk_100
	, ap.Walk_100 as away_pitcher_Walk_100
	, hp.WHIP_100 as home_pitcher_WHIP_100
	, ap.WHIP_100 as away_pitcher_WHIP_100
	, hp.Strikeout_100 as home_pitcher_Strikeout_100
	, ap.Strikeout_100 as away_pitcher_Strikeout_100
	, hp.Hit_By_Pitch_100 as home_pitcher_Hit_By_Pitch_100
	, ap.Hit_By_Pitch_100 as away_pitcher_Hit_By_Pitch_100
	, hp.num_game as home_pitcher_num_game_100
	, ap.num_game as away_pitcher_num_game_100
	, pg.home_throwinghand
	, pg.away_throwinghand
	, pg.home_streak as home_team_streak
	, pg.away_streak as away_team_streak
	, pg.home_wins as home_pitcher_season_wins
	, pg.home_losses as home_pitcher_season_losses
	, pg.away_wins as away_pitcher_season_wins
	, pg.away_losses as away_pitcher_season_losses
	, pg.home_hits as home_pitcher_season_hits
	, pg.home_runs as home_pitcher_season_runs
	, pg.home_errors as home_pitcher_season_errors
	, pg.away_hits as away_pitcher_season_hits
	, pg.away_runs as away_pitcher_season_runs
	, pg.away_errors as away_pitcher_season_errors
	, po.home_best_odds
	, po.away_best_odds
from game g		/* correct this */
left join pitcher_100 hp on g.home_pitcher = hp.pitcher and hp.game_id = g.game_id
left join pitcher_100 ap on g.away_pitcher = ap.pitcher and ap.game_id = g.game_id
left join team_results tr on g.game_id = tr.game_id and g.home_team_id = tr.team_id
left join pregame_detail pg on g.game_id = pg.game_id
left join pregame_odds_best_final po on g.game_id = po.game_id
order by game asc
;

alter table game_features
add primary key (game_id);

select * from game_features;
