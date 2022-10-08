# may need to assign SPARK_CLASSPATH before running:
# export SPARK_CLASSPATH='/path/xxx.jar:/path/xx2.jar'

import sys

from pyspark.sql import SparkSession


def main():
    spark = (
        SparkSession.builder.appName("pyspark exanple project")
        .config(
            "spark.jars",
            "/Users/sean/workspace/Sean/SDSU/BDA602/DBs/mariadb-java-client-3.0.6.jar",
        )
        .getOrCreate()
    )

    df = spark.sql(
        """
        with
        batter_name as (
            select
                batter,
                SUBSTRING_INDEX(des, ' ', 2) as name
            from atbat_r
            group by batter),

        per_game_stats as (
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
            where substring_index(g.local_date, "-", 1) is not null
                and bn.name is not null
            group by id, date
            order by id asc, date desc)

        select
            p1.name as name,
            p1.id as id,
            p1.team as team,
            p1.teamid as teamid,
            p1.date as date,
            /* date diff needs restrictions on both ends of range, */
            /* plus case restriction to make sure the player id matches. */
            /* haven't tested what happens when a player switches teams */
            sum(case when datediff(p1.date, p2.date) between 1 and 100
                and p1.id = p2.id then p2.AB end) as 100_day_ab,
            sum(case when datediff(p1.date, p2.date) between 1
                and 100 and p1.id = p2.id then p2.hit end) as 100_day_hit,
            sum(case when datediff(p1.date, p2.date) between 1
                and 100 and p1.id = p2.id then p2.walk end) as 100_day_walk,
            sum(case when datediff(p1.date, p2.date) between 1
                and 100 and p1.id = p2.id then p2.HR end) as 100_day_HR,
            avg(case when datediff(p1.date, p2.date) between 1
                and 100 and p1.id = p2.id then p2.bavg end) as 100_day_bavg,
            sum(case when datediff(p1.date, p2.date) between 1
                and 100 and p1.id = p2.id then 1 end) as num_game
        FROM per_game_stats p1
        left join per_game_stats p2 on p1.name = p2.name
        group by p1.date, id
        order by p1.id desc, p1.date asc
        """
    )

    df.show()
    return


if __name__ == "__main__":
    sys.exit(main())
