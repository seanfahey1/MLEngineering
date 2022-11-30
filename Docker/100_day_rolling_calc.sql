use baseballdb;

create table if not exists batter_per_game (primary key(batter_id, game)) as
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

create table if not exists rolling_100 as
select
    ra1.game,
    ra1.local_date,
    ra1.batter_id,
    (select sum(Hit)
        from batter_per_game ra2
        where ra2.local_date > DATE_ADD(ra1.local_date, interval - 100 day) and
            ra2.local_date < ra1.local_date and ra1.batter_id = ra2.batter_id) as last_100_days_hits,
    (select sum(AB)
        from batter_per_game ra2
        where ra2.local_date > DATE_ADD(ra1.local_date, interval - 100 day) and
            ra2.local_date < ra1.local_date and ra1.batter_id = ra2.batter_id) as last_100_days_atbats
    from batter_per_game ra1
    left join batter_per_game ra2 on ra1.batter_id = ra2.batter_id
    where ra1.game = 12560
    group by ra1.local_date, ra1.batter_id
;

select * from baseballdb.rolling_100;
