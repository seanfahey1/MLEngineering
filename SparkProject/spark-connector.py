import sys

from pyspark import StorageLevel
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession


def main():
    appName = "baseball mariadb spark"
    master = "local"
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

    user = "root"
    password = "password"  # pragma: allowlist secret

    jdbc_url = "jdbc:mysql://localhost:3306/baseballdb?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    game = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "baseball.game")
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    game.createOrReplaceTempView("game")
    game.persist(StorageLevel.DISK_ONLY)

    batter_counts = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "baseball.batter_counts")
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    batter_counts.createOrReplaceTempView("batter_counts")
    batter_counts.persist(StorageLevel.DISK_ONLY)

    atbat_r = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "baseball.atbat_r")
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    atbat_r.createOrReplaceTempView("atbat_r")
    atbat_r.persist(StorageLevel.DISK_ONLY)

    team = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "baseball.team")
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    team.createOrReplaceTempView("team")
    team.persist(StorageLevel.DISK_ONLY)

    batter_name = spark.sql(
        """
        select
            batter,
            first(SUBSTRING_INDEX(des, ' ', 2)) as name
        from atbat_r
        group by batter
        ;
        """
    )
    batter_name.createOrReplaceTempView("batter_name")
    batter_name.persist(StorageLevel.DISK_ONLY)

    per_game_stats = spark.sql(
        """
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
        """
    )
    per_game_stats.createOrReplaceTempView("per_game_stats")
    per_game_stats.persist(StorageLevel.DISK_ONLY)

    query = """
        select
            p1.name as name,
            p1.id as id,
            p1.team as team,
            p1.teamid as teamid,
            p1.date as date,
            /* date diff needs restrictions on both ends of range, /*
            /* plus case restriction to make sure the player id matches. */
            /* haven't tested what happens when a player switches teams (id id changes or not) */
            sum(case when datediff(p1.date, p2.date) between 1 and 100
                and p1.id = p2.id then p2.AB end) as 100_day_ab,
            sum(case when datediff(p1.date, p2.date) between 1 and 100
                and p1.id = p2.id then p2.hit end) as 100_day_hit,
            sum(case when datediff(p1.date, p2.date) between 1 and 100
                and p1.id = p2.id then p2.walk end) as 100_day_walk,
            sum(case when datediff(p1.date, p2.date) between 1 and 100
                and p1.id = p2.id then p2.HR end) as 100_day_HR,
            avg(case when datediff(p1.date, p2.date) between 1 and 100
                and p1.id = p2.id then p2.bavg end) as 100_day_bavg,
            sum(case when datediff(p1.date, p2.date) between 1 and 100
                and p1.id = p2.id then 1 end) as num_game
        FROM per_game_stats p1
        left join per_game_stats p2 on p1.name = p2.name
        group by p1.date, id
        order by p1.id desc, p1.date asc
    ;
    """

    df = SQLTransformer().setStatement(query)
    df.transform(per_game_stats).show()


if __name__ == "__main__":
    sys.exit(main())
