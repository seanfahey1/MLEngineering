#!/usr/bin/env python3
import sys

from pyspark import StorageLevel
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession


def main():
    app_name = "baseball mariadb spark"
    master = "local"
    jar_path = (
        "/Users/sean/workspace/Sean/sdsu/BDA602/DBs/mariadb-java-client-3.0.8.jar"
    )
    spark = (
        SparkSession.builder.config("spark.jars", jar_path)
        .appName(app_name)
        .master(master)
        .getOrCreate()
    )

    user = "sean"
    password = input("enter password...\t")  # pragma: allowlist secret

    jdbc_url = "jdbc:mysql://localhost:3306/baseballdb?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"
    dbtable = "baseballdb.game_features"

    game_features = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("driver", jdbc_driver)
        .option("user", user)
        .option("password", password)
        .option("dbtable", dbtable)
        .load()
    )

    game_features.createOrReplaceTempView("game_features_pyspark")
    game_features.persist(StorageLevel.DISK_ONLY)

    query = "select * from game_features_pyspark;"

    df = SQLTransformer().setStatement(query)
    df.transform(game_features).show()
    df = df.transform(game_features).toPandas()
    print(len(df))
    print(len(df.dropna(axis=0)))


if __name__ == "__main__":
    sys.exit(main())
