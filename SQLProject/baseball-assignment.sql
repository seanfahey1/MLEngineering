-- setting up a player name view b/c what good is any of this if we don't know the player's we're looking at
create or replace view batter_names as
	select 
		batter, 
		SUBSTRING_INDEX(des, ' ', 2) as name 
	from atbat_r 
	group by batter
;

-- career batting average calculion
select 
	bc.batter as id, 
	bn.name, 
	bc.team_id as team, 
	sum(bc.atBat) as AB, 
	sum(bc.Hit) as hits, 
	sum(bc.Walk) as walks, 
	sum(bc.Home_Run) as HR, 
	sum(bc.Hit) / sum(bc.atBat) as bavg 
from batter_counts bc 
left join batter_names bn on bc.batter=bn.batter 
group by bc.batter 
order by hits desc 
;


-- batting average calc per year
select 
	bc.batter as id, 
	bn.name, 
	bc.team_id as team, 
	sum(bc.atBat) as AB, 
	sum(bc.Hit) as hits, 
	sum(bc.Walk) as walks, 
	sum(bc.Home_Run) as HR, 
	sum(bc.Hit) / sum(bc.atBat) as bavg, 
	substring_index(g.local_date, "-", 1) as year
from batter_counts bc 
left join game g on g.game_id = bc.game_id
left join batter_names bn on bc.batter=bn.batter 
where substring_index(g.local_date, "-", 1) is not null
group by bc.batter, year
order by hits desc
;