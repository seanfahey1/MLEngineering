#!/usr/bin/env python3
import itertools
import pickle as p
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.io import to_html
from pyspark import StorageLevel
from pyspark.ml.feature import SQLTransformer
from pyspark.sql import SparkSession

sys.path.append(str(Path(__file__).parent / "../midterm"))
import midterm  # noqa: E402

warnings.filterwarnings("ignore")


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

    with open(Path(__file__).parent / "local_copy.p", "wb") as file:
        p.dump(df, file)

    return df


def cat_v_cat_predictor_correlations(
    df,
    categorical_predictors,
    response,
    output_dir,
    html_predictor_comparison_table_output_file,
):
    with open(html_predictor_comparison_table_output_file, "w") as out:
        out.write("")

    # set up empty dfs to store outputs
    brute_force_table_cat_cat = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "predictor 1 type",
            "predictor 2 type",
            "Difference of Mean Response",
            "Weighted DMR",
            "Population Proportion Plot",
            "Average Response Plot",
        ]
    )
    correlation_table_cat_cat = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "predictor 1 type",
            "predictor 2 type",
            "correlation ratio (Cramer's V)",
            "link to heat plot",
        ]
    )
    cat_cat_correlation_array = pd.DataFrame(
        columns=categorical_predictors, index=categorical_predictors
    )
    # get all pairwise unique combinations of categorical predictors
    for catpred1, catpred2 in itertools.combinations(categorical_predictors, 2):
        # cat. v. cat. correlation ratio
        corr = midterm.cat_correlation(df[catpred1], df[catpred2])
        cat_cat_correlation_array[catpred1][catpred2] = abs(corr)
        cat_cat_correlation_array[catpred2][catpred1] = abs(corr)

        midterm.plot_cat_cat(df, catpred1, catpred2, output_dir)

        correlation_table_cat_cat.loc[len(correlation_table_cat_cat)] = [
            catpred1,
            catpred2,
            "categorical",
            "categorical",
            corr,
            f"<a href='//{output_dir}/predictor v. predictor correlation plot - {catpred1} v. {catpred2}.html'>correlation plot link</a>",  # noqa: E501
        ]
        # cat. v. cat. wDMR
        dmr, wdmr = midterm.dmr_2d_cat_cat(df, catpred1, catpred2, response, output_dir)
        brute_force_table_cat_cat.loc[len(brute_force_table_cat_cat)] = [
            catpred1,
            catpred2,
            "categorical",
            "categorical",
            dmr,
            wdmr,
            f"<a href='//{output_dir}/predictor v. predictor population proportion - {catpred1} v. {catpred2}.html'>population proportion link</a>",  # noqa: E501
            f"<a href='//{output_dir}/predictor v. predictor average response - {catpred1} v. {catpred2}.html'>average response link</a>",  # noqa: E501
        ]

    brute_force_table_cat_cat.sort_values(
        by="Weighted DMR", ascending=False, inplace=True
    )
    correlation_table_cat_cat.sort_values(
        by="correlation ratio (Cramer's V)", ascending=False, inplace=True
    )

    for x in cat_cat_correlation_array.columns:
        cat_cat_correlation_array[x][x] = 1

    cat_cat_correlation_plot = px.imshow(cat_cat_correlation_array, text_auto=True)
    cat_cat_correlation_plot.update_layout(
        title="Cramer's V correlation metric between predictors",
        xaxis_title="categorical predictors",
        yaxis_title="categorical predictors",
    )

    # write html table outputs
    with open(html_predictor_comparison_table_output_file, "a") as out:
        out.write("<b><h1>Predictor v. Predictor Comparisons</h1></b>")
        out.write("<h1>Categorical-Categorical Predictor Comparisons</h1>")
        out.write("<h3>Weighted Difference of Mean Response</h3>")
        out.write(
            brute_force_table_cat_cat.to_html(
                index=False,
                na_rep="None",
                render_links=True,
                escape=False,
            )
        )
        out.write("<br><h3>Correlation Heat Plots</h3>")
        out.write(
            correlation_table_cat_cat.to_html(
                index=False,
                na_rep="None",
                render_links=True,
                escape=False,
            )
        )
        out.write("<br>")
        out.write(to_html(cat_cat_correlation_plot, include_plotlyjs="cdn"))


def cat_v_cont_predictor_correlations(
    df,
    categorical_predictors,
    continuous_predictors,
    response,
    output_dir,
    html_predictor_comparison_table_output_file,
):
    # set up empty dfs to store outputs
    brute_force_table_cat_cont = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "predictor 1 type",
            "predictor 2 type",
            "Difference of Mean Response",
            "Weighted DMR",
            "Population Proportion Plot",
            "Average Response Plot",
        ]
    )
    correlation_table_cat_cont = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "predictor 1 type",
            "predictor 2 type",
            "correlation ratio",
            "link to violin plot",
        ]
    )
    cat_cont_correlation_array = pd.DataFrame(
        columns=categorical_predictors, index=continuous_predictors
    )
    # get all combinations between categorical and continuous predictors
    for catpred in categorical_predictors:
        for contpred in continuous_predictors:
            # cat. v. cont. correlation ratio
            corr = midterm.cat_cont_correlation_ratio(df[catpred], df[contpred])
            cat_cont_correlation_array[catpred][contpred] = abs(corr)

            midterm.plot_cat_cont(df, catpred, contpred, output_dir)

            correlation_table_cat_cont.loc[len(correlation_table_cat_cont)] = [
                catpred,
                contpred,
                "categorical",
                "continuous",
                corr,
                f"<a href='//{output_dir}/predictor v. predictor correlation plot - {catpred} v. {contpred}.html'>correlation plot link</a>",  # noqa: E501
            ]
            # cat. v. cont. wDMR
            dmr, wdmr = midterm.dmr_2d_cat_cont(
                df, catpred, contpred, response, output_dir
            )
            brute_force_table_cat_cont.loc[len(brute_force_table_cat_cont)] = [
                catpred,
                contpred,
                "categorical",
                "continuous",
                dmr,
                wdmr,
                f"<a href='//{output_dir}/predictor v. predictor population proportion - {catpred} v. {contpred}.html'>population proportion link</a>",  # noqa: E501
                f"<a href='//{output_dir}/predictor v. predictor average response - {catpred} v. {contpred}.html'>average response link</a>",  # noqa: E501
            ]

    brute_force_table_cat_cont.sort_values(
        by="Weighted DMR", ascending=False, inplace=True
    )
    correlation_table_cat_cont.sort_values(
        by="correlation ratio", ascending=False, inplace=True
    )

    cat_cont_correlation_plot = px.imshow(cat_cont_correlation_array, text_auto=True)
    cat_cont_correlation_plot.update_layout(
        title="Correlation Ratio between categorical and continuous predictors",
        xaxis_title="categorical predictors",
        yaxis_title="continuous predictors",
    )

    # write html table outputs
    with open(html_predictor_comparison_table_output_file, "a") as out:
        out.write("<br><br><h1>Categorical-Continuous Predictor Comparisons</h1>")
        out.write("<h3>Weighted Difference of Mean Response</h3>")
        out.write(
            brute_force_table_cat_cont.to_html(
                index=False,
                na_rep="None",
                render_links=True,
                escape=False,
            )
        )
        out.write("<br><h3>Correlation Violin Plots</h3>")
        out.write(
            correlation_table_cat_cont.to_html(
                index=False,
                na_rep="None",
                render_links=True,
                escape=False,
            )
        )
        out.write("<br>")
        out.write(to_html(cat_cont_correlation_plot, include_plotlyjs="cdn"))


def cont_v_cont_predictor_correlations(
    df,
    continuous_predictors,
    response,
    output_dir,
    html_predictor_comparison_table_output_file,
):
    # set up empty dfs to store outputs
    brute_force_table_cont_cont = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "predictor 1 type",
            "predictor 2 type",
            "Difference of Mean Response",
            "Weighted DMR",
            "Population Proportion Plot",
            "Average Response Plot",
        ]
    )
    correlation_table_cont_cont = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "predictor 1 type",
            "predictor 2 type",
            "Pearson's Ratio",
            "Pearson's absolute value",
            "p-value",
            "t-value",
            "link to scatter plot",
        ]
    )
    cont_cont_correlation_array = pd.DataFrame(
        columns=continuous_predictors, index=continuous_predictors
    )
    for contpred1, contpred2 in itertools.combinations(continuous_predictors, 2):
        # continuous v. continuous correlation ratio
        p_value, t_value, corr = midterm.linear_regression(df, contpred1, contpred2)

        cont_cont_correlation_array[contpred1][contpred2] = abs(corr)
        cont_cont_correlation_array[contpred2][contpred1] = abs(corr)

        midterm.plot_cont_cont(df, contpred1, contpred2, output_dir)

        correlation_table_cont_cont.loc[len(correlation_table_cont_cont)] = [
            contpred1,
            contpred2,
            "continuous",
            "continuous",
            corr,
            abs(corr),
            p_value,
            t_value,
            f"<a href='//{output_dir}/predictor v. predictor correlation plot - {contpred1} v. {contpred2}.html'>correlation plot link</a>",  # noqa: E501
        ]
        # cont. v. cont. wDMR
        dmr, wdmr = midterm.dmr_2d_cont_cont(
            df, contpred1, contpred2, response, output_dir
        )
        brute_force_table_cont_cont.loc[len(brute_force_table_cont_cont)] = [
            contpred1,
            contpred2,
            "continuous",
            "continuous",
            dmr,
            wdmr,
            f"<a href='//{output_dir}/predictor v. predictor population proportion - {contpred1} v. {contpred2}.html'>population proportion link</a>",  # noqa: E501
            f"<a href='//{output_dir}/predictor v. predictor average response - {contpred1} v. {contpred2}.html'>average response link</a>",  # noqa: E501
        ]

    brute_force_table_cont_cont.sort_values(
        by="Weighted DMR", ascending=False, inplace=True
    )
    correlation_table_cont_cont.sort_values(
        by="Pearson's absolute value", ascending=False, inplace=True
    )

    for x in cont_cont_correlation_array.columns:
        cont_cont_correlation_array[x][x] = 1

    cont_cont_correlation_plot = px.imshow(cont_cont_correlation_array, text_auto=True)
    cont_cont_correlation_plot.update_layout(
        title="R correlation metric between predictors",
        xaxis_title="continuous predictors",
        yaxis_title="continuous predictors",
    )

    # write html table outputs
    with open(html_predictor_comparison_table_output_file, "a") as out:
        out.write("<br><br><h1>Continuous-Continuous Predictor Comparisons</h1>")
        out.write("<h3>Weighted Difference of Mean Response</h3>")
        out.write(
            brute_force_table_cont_cont.to_html(
                index=False,
                na_rep="None",
                render_links=True,
                escape=False,
            )
        )
        out.write("<br><h3>Correlation Scatter Plots</h3>")
        out.write(
            correlation_table_cont_cont.to_html(
                index=False,
                na_rep="None",
                render_links=True,
                escape=False,
            )
        )
        out.write("<br>")
        out.write(to_html(cont_cont_correlation_plot, include_plotlyjs="cdn"))


def predictor_v_response_correlations(
    df,
    categorical_predictors,
    continuous_predictors,
    response,
    output_dir,
    html_predictor_comparison_table_output_file,
):
    output_df = pd.DataFrame(
        columns=[
            "Response",
            "Predictor",
            "Response Type",
            "Predictor Type",
            "predictor min",
            "predictor max",
            "predictor median",
            "correlation metric",
            "correlation type",
            "RF importance",
            "DMR",
            "wDMR",
            "link to DMR plot",
            "link to predictor plot",
        ]
    )

    # Start with continuous predictors
    feature_importances = midterm.random_forest(
        df, response, continuous_predictors, "categorical"
    )

    for i, predictor in enumerate(continuous_predictors):
        midterm.plot_cont_pred_cat_resp(df, predictor, response, output_dir)
        corr = midterm.cat_cont_correlation_ratio(df[response], df[predictor])
        corr_type = "Correlation Ratio"

        msd_df, dmr, wdmr = midterm.difference_w_mean_of_resp_1d_cont_pred(
            df, predictor, response, output_dir
        )

        output_df.loc[len(output_df)] = [
            response,
            predictor,
            "categorical",
            "continuous",
            min(df[predictor]),
            max(df[predictor]),
            np.median(df[predictor]),
            corr,
            corr_type,
            feature_importances[i],
            dmr,
            wdmr,
            f"<a href='//{output_dir}/msd plot {predictor} v. response ({response}).html'>DMR {predictor} link</a>",  # noqa: E501
            f"<a href='//{output_dir}/{predictor} v. response.html'>{predictor} v. response link</a>",  # noqa: E501
        ]

    # Then add categorical predictors
    for i, predictor in enumerate(categorical_predictors):
        midterm.plot_cat_pred_cat_resp(df, predictor, response, output_dir)
        corr = midterm.cat_correlation(df[predictor], df[response])
        corr_type = "Cramer's V"

        msd_df, dmr, wdmr = midterm.difference_w_mean_of_resp_1d_cat_pred(
            df, predictor, response, output_dir
        )

        output_df.loc[len(output_df)] = [
            response,
            predictor,
            "categorical",
            "categorical",
            None,
            None,
            None,
            corr,
            corr_type,
            None,
            dmr,
            wdmr,
            f"<a href='//{output_dir}/msd plot {predictor} v. response ({response}).html'>DMR {predictor} link</a>",  # noqa: E501
            f"<a href='//{output_dir}/{predictor} v. response.html'>{predictor} v. response link</a>",  # noqa: E501
        ]

    # write output table
    output_df.sort_values(by="wDMR", ascending=False, inplace=True)
    with open(html_predictor_comparison_table_output_file, "a") as out:
        out.write("<b><h1>Predictor v. Response Comparisons</h1></b>")
        out.write("<h3>Weighted Difference of Mean Response Table</h3>")
        out.write(
            output_df.to_html(
                index=False,
                na_rep="None",
                render_links=True,
                escape=False,
            )
        )


def main(load_from_disk=False):
    # setup output directory
    cwd = Path(__file__).parent.resolve()
    output_dir = cwd / "output"
    Path(output_dir).mkdir(exist_ok=True)

    if Path(Path(__file__).parent / "local_copy.p").is_file() and load_from_disk:
        print("Found file locally. Loading file from disk.")
        with open(Path(__file__).parent / "local_copy.p", "rb") as file:
            df = p.load(file)
    else:
        jar_path = (
            "/Users/sean/workspace/Sean/sdsu/BDA602/DBs/mariadb-java-client-3.0.8.jar"
        )
        user = "sean"
        dbtable = "baseballdb.game_features"
        query = "select * from game_features_pyspark;"
        password = input("enter password...\t")  # pragma: allowlist secret
        df = baseballdb_connection(user, password, jar_path, dbtable, query)

    print("Cleaning up DataFrame")
    convert_to_floats = [
        "home_win_rate",
        "away_win_rate",
        "home_pitcher_atBat_100",
        "away_pitcher_atBat_100",
        "home_pitcher_plateApperance_100",
        "away_pitcher_plateApperance_100",
        "home_pitcher_Hit_100",
        "away_pitcher_Hit_100",
        "home_pitcher_num_game_100",
        "away_pitcher_num_game_100",
    ]
    convert_to_string = ["home_throwinghand", "away_throwinghand"]

    for column in convert_to_floats:
        df[column] = df[column].apply(pd.to_numeric)

    for column in convert_to_string:
        df[column] = [
            np.nan if str(x) == "None" or x == "" else str(x)
            for x in df[column].tolist()
        ]
    # df[convert_to_string]
    # df[convert_to_string] = df[convert_to_string].astype(str)

    categorical_predictors = ["home_throwinghand", "away_throwinghand"]
    continuous_predictors = [
        "home_win_rate",
        "away_win_rate",
        "home_pitcher",
        "away_pitcher",
        "home_pitcher_atBat_100",
        "away_pitcher_atBat_100",
        "home_pitcher_plateApperance_100",
        "away_pitcher_plateApperance_100",
        "home_pitcher_Hit_100",
        "away_pitcher_Hit_100",
        "home_pitcher_Single_100",
        "away_pitcher_Single_100",
        "home_pitcher_Double_100",
        "away_pitcher_Double_100",
        "home_pitcher_Triple_100",
        "away_pitcher_Triple_100",
        "home_pitcher_Home_Run_100",
        "away_pitcher_Home_Run_100",
        "home_pitcher_Walk_100",
        "away_pitcher_Walk_100",
        "home_pitcher_WHIP_100",
        "away_pitcher_WHIP_100",
        "home_pitcher_Strikeout_100",
        "away_pitcher_Strikeout_100",
        "home_pitcher_Hit_By_Pitch_100",
        "away_pitcher_Hit_By_Pitch_100",
        "home_pitcher_num_game_100",
        "away_pitcher_num_game_100",
        "home_team_streak",
        "away_team_streak",
        "home_pitcher_season_wins",
        "home_pitcher_season_losses",
        "away_pitcher_season_wins",
        "away_pitcher_season_losses",
        "home_pitcher_season_hits",
        "home_pitcher_season_runs",
        "home_pitcher_season_errors",
        "away_pitcher_season_hits",
        "away_pitcher_season_runs",
        "away_pitcher_season_errors",
        "home_best_odds",
        "away_best_odds",
    ]
    response = "home_team_wins"

    # PREDICTOR V. PREDICTOR CORRELATIONS
    print("Predictor v. predictor correlations")
    # setup empty html files to build onto
    html_predictor_comparison_table_output_file = f"{output_dir}/comparison-tables.html"

    # categorical v. categorical predictors
    cat_v_cat_predictor_correlations(
        df,
        categorical_predictors,
        response,
        output_dir,
        html_predictor_comparison_table_output_file,
    )

    # categorical v. continuous predictors
    cat_v_cont_predictor_correlations(
        df,
        categorical_predictors,
        continuous_predictors,
        response,
        output_dir,
        html_predictor_comparison_table_output_file,
    )

    # continuous v. continuous predictors
    cont_v_cont_predictor_correlations(
        df,
        continuous_predictors,
        response,
        output_dir,
        html_predictor_comparison_table_output_file,
    )

    # PREDICTOR V. RESPONSE CORRELATIONS
    print("Predictor v. response correlations")
    predictor_v_response_correlations(
        df,
        categorical_predictors,
        continuous_predictors,
        response,
        output_dir,
        html_predictor_comparison_table_output_file,
    )


if __name__ == "__main__":
    sys.exit(main(True))
