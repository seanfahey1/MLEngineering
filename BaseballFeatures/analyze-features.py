#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd
from pyspark import StorageLevel
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession

HERE = Path(__file__).parent
sys.path.append(str(HERE / "../midterm"))
import midterm  # noqa: E402


def baseballdb_connection(
    user: str, password: str, jar_path: str, dbtable: str, query: str
) -> pd.DataFrame:

    app_name = "baseball mariadb spark"
    master = "local"
    jdbc_url = "jdbc:mysql://localhost:3306/baseballdb?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    # initialize spark and set logging level to suppress warnings
    spark = (
        SparkSession.builder.config("spark.jars", jar_path)
        .appName(app_name)
        .master(master)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("FATAL")

    table = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("driver", jdbc_driver)
        .option("user", user)
        .option("password", password)
        .option("dbtable", dbtable)
        .load()
    )

    table.createOrReplaceTempView("game_features_pyspark")
    table.persist(StorageLevel.DISK_ONLY)

    df = SQLTransformer().setStatement(query)
    # df.transform(game_features).show()
    df = df.transform(table).select("*").toPandas()

    return df


def main():
    jar_path = (
        "/Users/sean/workspace/Sean/sdsu/BDA602/DBs/mariadb-java-client-3.0.8.jar"
    )
    user = "sean"
    password = input("enter password...\t")  # pragma: allowlist secret
    dbtable = "baseballdb.game_features"
    query = "select * from game_features_pyspark;"

    df = baseballdb_connection(user, password, jar_path, dbtable, query)

    midterm.difference_w_mean_of_resp_1d_cat_pred(
        df, "away_pitcher_WHIP_100", "home_team_wins", Path(".")
    )


if __name__ == "__main__":
    sys.exit(main())
