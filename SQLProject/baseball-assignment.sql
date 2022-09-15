-- setting up a player name view b/c what good is any of this if we don't know the player's we're looking at
create or replace view batter_names as
	select 
		batter, 
		SUBSTRING_INDEX(des, ' ', 2) as name 
	from atbat_r 
	group by batter
;

-- career batting average calculion
create or replace table career_bavg
	select 
		bc.batter as id, 
		bn.name, 
		bc.team_id as teamid, 
		t.name_brief as team,
		sum(bc.atBat) as AB, 
		sum(bc.Hit) as hits, 
		sum(bc.Walk) as walks, 
		sum(bc.Home_Run) as HR, 
		case when sum(bc.atBat) = 0 then 0 else sum(bc.Hit) / sum(bc.atBat) end as bavg
	from batter_counts bc 
	left join batter_names bn on bc.batter=bn.batter
	left join team t on t.team_id = bc.team_id
	group by bc.batter 
	order by hits desc 
;


-- batting average calc per year
create or replace table year_bavg
	select 
		bc.batter as id, 
		bn.name, 
		bc.team_id as teamid, 
		t.name_brief as team,
		sum(bc.atBat) as AB, 
		sum(bc.Hit) as hits, 
		sum(bc.Walk) as walks, 
		sum(bc.Home_Run) as HR, 
		case when sum(bc.atBat) = 0 then 0 else sum(bc.Hit) / sum(bc.atBat) end as bavg,
		substring_index(g.local_date, "-", 1) as year
	from batter_counts bc 
	left join game g on g.game_id = bc.game_id
	left join batter_names bn on bc.batter=bn.batter 
	left join team t on t.team_id = bc.team_id
	where substring_index(g.local_date, "-", 1) is not null /* remove null dates */
	group by bc.batter, year
	order by hits desc
;

-- rolling 100 day average
-- intermediate table of player stats per game
create or replace table per_game_stats as 
	select 
		bc.batter as id, 
		bn.name, 
		bc.team_id as teamid, 
		t.name_brief as team,
		g.`game_id`as game,
		sum(bc.atBat) as AB, 
	 	sum(bc.Hit) as hits, 
	 	sum(bc.Walk) as walks, 
		sum(bc.Home_Run) as HR, 
		case when sum(bc.atBat) = 0 then 0 else sum(bc.Hit) / sum(bc.atBat) end as bavg,
		cast(g.local_date as date) as date
	from batter_counts bc 
	left join game g on g.game_id = bc.game_id
	left join batter_names bn on bc.batter=bn.batter
	left join team t on t.team_id = bc.team_id
	where substring_index(g.local_date, "-", 1) is not null /* remove null dates */
	group by id, date
	order by id asc, date desc
;

-- get rolling average table
select 
	name,
	id,
	team,
	teamid,
	date,
	sum(AB) over(order by date rows between 99 preceding and current row) as 100_game_AB,
	sum(hits) over(order by date rows between 99 preceding and current row) as 100_game_hits,
	avg(bavg) over(order by date rows between 99 preceding AND current row) as 100_game_avg
FROM per_game_stats
where name is not null
order by id asc, date asc
;
