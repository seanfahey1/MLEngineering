-- startup/cleanup tasks
drop view if exists batter_name;
drop table if exists career_bavg;
drop table if exists year_bavg;
drop table if exists per_game_stats;
drop table if exists rolling_bavg;


-- setting up a player name view b/c what good is any of this if we don't know the names of player's that we're looking at
create or replace view batter_name as
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
		sum(bc.Hit) as hit, 
		sum(bc.Walk) as walk, 
		sum(bc.Home_Run) as HR, 
		case when sum(bc.atBat) = 0 then 0 else sum(bc.Hit) / sum(bc.atBat) end as bavg
	from batter_counts bc 
	left join batter_name bn on bc.batter=bn.batter
	left join team t on t.team_id = bc.team_id
	group by bc.batter 
	order by hit desc 
;


-- per year batting average calculation
create or replace table year_bavg
	select 
		bc.batter as id, 
		bn.name, 
		bc.team_id as teamid, 
		t.name_brief as team,
		sum(bc.atBat) as AB, 
		sum(bc.Hit) as hit, 
		sum(bc.Walk) as walk, 
		sum(bc.Home_Run) as HR, 
		case when sum(bc.atBat) = 0 then 0 else sum(bc.Hit) / sum(bc.atBat) end as bavg,
		substring_index(g.local_date, "-", 1) as year
	from batter_counts bc 
	left join game g on g.game_id = bc.game_id
	left join batter_name bn on bc.batter=bn.batter 
	left join team t on t.team_id = bc.team_id
	where substring_index(g.local_date, "-", 1) is not null /* remove null dates */
	group by bc.batter, year
	order by hit desc
;

-- rolling 100 day average: 2 step process
-- intermediate table of player stats per game
create or replace table per_game_stats as 
	select 
		row_number() over(order by id asc) as idx,
		bc.batter as id, 
		bn.name, 
		bc.team_id as teamid, 
		t.name_brief as team,
		g.`game_id`as game,
		bc.atBat as AB, 
	 	bc.Hit as hit, 
	 	bc.Walk as walk, 
		bc.Home_Run as HR, 
		case when bc.atBat = 0 then 0 else bc.Hit / bc.atBat end as bavg,
		cast(g.local_date as date) as date
	from batter_counts bc 
	left join game g on g.game_id = bc.game_id
	left join batter_name bn on bc.batter=bn.batter
	left join team t on t.team_id = bc.team_id
	where substring_index(g.local_date, "-", 1) is not null /* remove null dates */ and bn.name is not null
	group by id, date
	order by id asc, date desc
;

-- add a primary key to the above table
alter table per_game_stats add primary key (idx);


-- 100 day rolling average calculation
create or replace table rolling_bavg as
	select 
		p1.name as name,
		p1.id as id,
		p1.team as team,
		p1.teamid as teamid,
		p1.date as date,
		/* date diff needs restrictions on both ends of range, plus case restriction to make sure the player id matches. */
		/* haven't tested what happens when a player switches teams (id id changes or not) */
		sum(case when datediff(p1.date, p2.date) between 1 and 100 and p1.id = p2.id then p2.AB end) as 100_day_ab,
		sum(case when datediff(p1.date, p2.date) between 1 and 100 and p1.id = p2.id then p2.hit end) as 100_day_hit,
		sum(case when datediff(p1.date, p2.date) between 1 and 100 and p1.id = p2.id then p2.walk end) as 100_day_walk,
		sum(case when datediff(p1.date, p2.date) between 1 and 100 and p1.id = p2.id then p2.HR end) as 100_day_HR,
		avg(case when datediff(p1.date, p2.date) between 1 and 100 and p1.id = p2.id then p2.bavg end) as 100_day_bavg,
		sum(case when datediff(p1.date, p2.date) between 1 and 100 and p1.id = p2.id then 1 end) as num_game
	FROM per_game_stats p1
	left join per_game_stats p2 on p1.name = p2.name 
	group by p1.date, id
	order by p1.id desc, p1.date asc
;

-- remove null values from the table (when a player appears in a game but hasn't played a game in the 100 days prior)
delete from rolling_bavg where num_game is null;


-- test on a single player
select * from rolling_bavg 
where id = "400085"
;
