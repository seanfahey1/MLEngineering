import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api
from plotly.io import to_html
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def determine_continuous(df, predictor):
    """
    Easy way to test if a variable is continuous for this assignment. We're only expecting boolean or continuous
    values, so drop NaN values and see how many unique values we actually have. If 2, then probably boolean.
    """
    values = set(df[predictor].dropna())
    if len(values) == 2:
        return False
    else:
        return True


def inspect_cont_pred_cont_resp(df, predictor, response, p_value, t_value):
    """
    Generates scatter plot with linear regression line for CONTINUOUS predictor and CONTINUOUS response.
    """
    fig = px.scatter(
        x=df[predictor],
        y=df[response],
        trendline="ols",
        title=f"<b>{predictor} v. {response}</b><br>continuous v. continuous<br>"
        f"<i>p-value: {p_value}, t-value: {t_value}</i>",
    )
    fig.update_layout(
        xaxis_title=predictor,
        yaxis_title=response,
    )
    # fig.show()
    return to_html(fig, include_plotlyjs="cdn")


def inspect_bool_pred_cont_resp(
    df, predictor, response, p_value, t_value, reverse=False
):
    """
    Generates violin plot grouped by the predictor when the predictor is CATEGORICAL/BOOLEAN and the response is
    CONTINUOUS. This function also works for the reverse (switching predictor and response) when reverse is set to
    True.
    """
    if reverse:
        cat = response
        cont = predictor
        title_text = f"continuous v. categorical<br><i>p-value: {p_value}, t-value: {t_value}</i>"
    else:
        cat = predictor
        cont = response
        title_text = f"categorical v. continuous<br><i>p-value: {p_value}, t-value: {t_value}</i>"

    fig = px.violin(
        color=df[cat],
        y=df[cont],
        points="all",
        box=True,
        title=f"<b>{cat} v. {cont}</b><br>{title_text}",
    )
    fig.update_layout(
        xaxis_title=cat,
        yaxis_title=cont,
    )
    # fig.show()
    return to_html(fig, include_plotlyjs="cdn")


def inspect_bool_pred_bool_resp(df, predictor, response, p_value, t_value):
    """
    Generates a heat map showing the frequency of each response value relative to each predictor value when both the
    predictor and response are CATEGORICAL/BOOLEAN.
    """
    # TODO: should this be normalized by the predictor?? Probably?
    count_df = (
        df[[predictor, response]]
        .value_counts()
        .reset_index()
        .pivot(index=predictor, columns=response, values=0)
    )
    fig = px.imshow(count_df)
    fig.update_layout(
        title=f"<b>{predictor} v. {response}</b><br>categorical v. categorical<br>"
        f"<i>p-value: {p_value}, t-value: {t_value}</i>",
        xaxis_title=response,
        yaxis_title=predictor,
    )
    # fig.show()
    return to_html(fig, include_plotlyjs="cdn")


def do_regression(df, predictor, response, method):
    # drop any NaN values
    df = df[[predictor, response]].dropna()

    # do p and t calculations
    if method == "linear":
        pred = statsmodels.api.add_constant(df[predictor])
        fitted_output = statsmodels.api.OLS(df[response], pred).fit(disp=0)
        t_value = round(fitted_output.tvalues[1], 6)
        p_value = "{:.6e}".format(fitted_output.pvalues[1])

    elif method == "logistic":
        fitted_output = statsmodels.api.Logit(df[predictor], df[response]).fit(disp=0)
        t_value = round(fitted_output.tvalues[0], 6)
        p_value = "{:.6e}".format(fitted_output.pvalues[0])

    elif method == "logistic-reversed":
        fitted_output = statsmodels.api.Logit(df[response], df[predictor]).fit(disp=0)
        t_value = round(fitted_output.tvalues[0], 6)
        p_value = "{:.6e}".format(fitted_output.pvalues[0])

    else:
        raise ValueError("Invalid method")

    return p_value, t_value


def weighted_mean_of_response(df, predictor, response):
    msd_df = pd.DataFrame(
        columns=[
            "(𝑖)",
            "LowerBin",
            "UpperBin",
            "BinCenters",
            "BinCount",
            "BinMeans (𝜇𝑖)",
            "PopulationMean (𝜇𝑝𝑜𝑝)",
            "MeanSquaredDiff",
            "PopulationProportion (𝑤𝑖)",
            "MeanSquaredDiffWeighted",
        ]
    )

    # get rid of NaN values and sort the dataframe
    df = df[[predictor, response]].dropna()
    df = df.sort_values(by=predictor, ascending=True).reset_index(drop=True)

    # figure out number of samples to use per bin
    bin_step_sizes = [((df[predictor].max() - df[predictor].min()) / 10)] * 9 + [np.inf]

    # calculate mean response per bin, bin predictor min, bin predictor max
    previous_bin_max = min(df[predictor])
    pop_mean_response = np.mean(df[response])

    for i, bin_step_size in enumerate(bin_step_sizes):
        bin_df = df[
            (df[predictor] >= previous_bin_max)
            & (df[predictor] < previous_bin_max + bin_step_size)
        ]
        bin_count = len(bin_df)
        pop_porp = bin_count / len(df)

        if bin_count > 0:
            bin_min = min(bin_df[predictor])
            bin_max = max(bin_df[predictor])
            bin_center = (bin_max - bin_min) / 2 + bin_min
            mean_response = np.mean(bin_df[response])
            msd = (mean_response - pop_mean_response) ** 2 / bin_count
            msd_weighted = msd / pop_porp

        else:
            bin_min = previous_bin_max
            bin_max = previous_bin_max + bin_step_size
            bin_center = (previous_bin_max + bin_step_size) / 2 + previous_bin_max
            mean_response = None
            msd = None
            msd_weighted = None

        msd_df.loc[len(msd_df)] = [
            str(int(i)),
            bin_min,
            bin_max,
            bin_center,
            bin_count,
            mean_response,
            pop_mean_response,
            msd,
            pop_porp,
            msd_weighted,
        ]
        previous_bin_max = bin_max
    msd_value = np.mean(msd_df["MeanSquaredDiff"])
    msd_weighted_value = np.mean(msd_df["MeanSquaredDiffWeighted"])

    msd_plot = plot_msd(msd_df, predictor)

    msd_df.loc[len(msd_df)] = (
        ["Averages"] + [""] * 6 + [str(msd_value)] + [""] + [str(msd_weighted_value)]
    )
    return msd_df.to_html(), msd_value, msd_weighted_value, msd_plot


def plot_msd(df, predictor):
    fig = go.Figure(
        layout=go.Layout(
            title="CLE vs Model",
            yaxis2=dict(overlaying="y", side="right"),
        )
    )
    fig.add_trace(
        go.Bar(name="counts", x=df["BinCenters"], y=df["BinCount"], yaxis="y1"),
    )
    fig.add_trace(
        go.Scatter(
            name="𝜇𝑝𝑜𝑝", x=df["BinCenters"], y=df["PopulationMean (𝜇𝑝𝑜𝑝)"], yaxis="y2"
        )
    )
    y = df["BinMeans (𝜇𝑖)"]
    print(y)
    fig.add_trace(
        go.Scatter(
            name="𝜇𝑝𝑜𝑝 - 𝜇𝑖",
            x=df["BinCenters"],
            y=y,
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=f"Binned difference with mean of response vs. Bin - {predictor}"
    )
    fig.update_layout(
        xaxis_title="Predictor Bin", yaxis_title="Population", yaxis2_title="Response"
    )
    fig.update_yaxes(rangemode="tozero")
    # fig.show()

    return to_html(fig, include_plotlyjs="cdn")


def get_dummies(df, predictor):
    uniques = df[predictor].unique()
    if set(df[predictor].to_list()) == {True, False}:
        conversion_dict = {True: 1, False: 0}
        new_column = [conversion_dict[x] for x in df[predictor].to_list()]
        new_column_name = f"bool_{predictor}"
        df[new_column_name] = new_column
        return df, True

    for value in uniques:
        if value not in [0, 1]:
            conversion_dict = {x: i for (i, x) in enumerate(uniques)}
            new_column = [conversion_dict[x] for x in df[predictor].to_list()]
            new_column_name = f"bool_{predictor}"
            df[new_column_name] = new_column
            return df, True
    return df, False


def random_forest(df, response, predictors, response_type):
    if response_type == "categorical":
        rf = RandomForestClassifier(random_state=0)
    elif response_type == "continuous":
        rf = RandomForestRegressor(random_state=0)
    else:
        raise ValueError("invalid response type")

    relevant_columns = [response] + predictors
    df = df[relevant_columns].dropna()
    X = df[predictors].to_numpy()

    if X.shape[1] == 1:
        X = X.reshape(-1, 1)
    y = df[response].to_numpy().reshape(-1, 1).ravel()
    rf.fit(X, y)
    return list(rf.feature_importances_)


def make_clickable(val, link):
    return '<a href="{}">{}</a>'.format(link, val)


def dataset_insert():
    """
    Function to easily drop in different csv formatted datasets and lists of predictor and response columns to use.
    """
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jorisvandenbossche/pandas-tutorial/master/data/titanic.csv"
    )
    df["test1"] = [random.choice([True, False]) for i in range(len(df))]
    df["test2"] = [random.choice([0, 1]) for i in range(len(df))]
    df["test3"] = [random.choice([x for x in range(100)]) for i in range(len(df))]
    predictors = ["Age", "test1", "test2", "test3", "Sex", "Survived", "Fare"]
    response = "test1"
    predictors = [x for x in predictors if x != response]
    # response = "Fare"
    return df, predictors, response


def main():
    # Using titanic dataset as placeholder for testing. Any pandas dataframe can be inserted here
    df, predictors, response = dataset_insert()

    response_continuous = determine_continuous(df, response)
    response_modified = False
    if not response_continuous:
        df, response_modified = get_dummies(df, response)

    output_df = pd.DataFrame(
        columns=[
            "Response",
            "Predictor",
            "Response Type",
            "Predictor Type",
            "p-value",
            "t-value",
            "DMR",
            "wDMR",
        ]
    )

    Path("./output").mkdir(exist_ok=True)

    continuous_predictors = []

    # loop through each predictor and generate the correct plot
    for predictor in predictors:
        print(predictor)
        # determine if predictor is categorical or continuous
        predictor_continuous = determine_continuous(df, predictor)
        modified = False

        if not predictor_continuous:
            df, modified = get_dummies(df, predictor)
        else:
            continuous_predictors.append(predictor)

        # use cat/cont info to determine which functions to call
        if response_continuous:
            if predictor_continuous:
                p_value, t_value = do_regression(
                    df, predictor, response, method="linear"
                )
                plot = inspect_cont_pred_cont_resp(
                    df, predictor, response, p_value, t_value
                )
                msd_html, dmr, wdmr, msd_plot = weighted_mean_of_response(
                    df, predictor, response
                )
            else:
                modified_predictor = f"bool_{predictor}" if modified else predictor
                p_value, t_value = do_regression(
                    df, modified_predictor, response, method="logistic"
                )
                plot = inspect_bool_pred_cont_resp(
                    df, predictor, response, p_value, t_value
                )
                msd_html, dmr, wdmr, msd_plot = weighted_mean_of_response(
                    df, modified_predictor, response
                )
        else:
            if predictor_continuous:
                modified_response = (
                    f"bool_{response}" if response_modified else response
                )
                p_value, t_value = do_regression(
                    df, predictor, modified_response, method="logistic-reversed"
                )
                plot = inspect_bool_pred_cont_resp(
                    df,
                    predictor,
                    response,
                    p_value,
                    t_value,
                    reverse=True,
                )
                msd_html, dmr, wdmr, msd_plot = weighted_mean_of_response(
                    df, predictor, modified_response
                )
            else:
                modified_predictor = f"bool_{predictor}" if modified else predictor
                modified_response = (
                    f"bool_{response}" if response_modified else response
                )
                p_value, t_value = do_regression(
                    df, modified_predictor, modified_response, method="logistic"
                )
                plot = inspect_bool_pred_bool_resp(
                    df, predictor, response, p_value, t_value
                )
                msd_html, dmr, wdmr, msd_plot = weighted_mean_of_response(
                    df, modified_predictor, modified_response
                )

        line = [
            response,
            predictor,
            "Continuous" if response_continuous else "Categorical",
            "Continuous" if predictor_continuous else "Categorical",
            p_value,
            t_value,
            dmr,
            wdmr,
        ]
        output_df.loc[len(output_df)] = line

        # save outputs
        with open(f"./output/{predictor}-wDMR-table.html", "w") as out:
            out.write(msd_html)
        with open(f"./output/{predictor}-DMR-plot.html", "w") as out:
            out.write(msd_plot)
        with open(f"./output/{predictor}-feature-plot.html", "w") as out:
            out.write(plot)

    # random forest feature importance calculations
    rf_results = random_forest(
        df,
        response,
        continuous_predictors,
        "continuous" if response_continuous else "categorical",
    )
    rf_dict = {continuous_predictors[i]: rf_results[i] for i in range(len(rf_results))}
    for predictor in predictors:
        if predictor not in rf_dict.keys():
            rf_dict[predictor] = None
    rf_list = [rf_dict[x] for x in predictors]
    output_df.insert(6, "RF Importance", rf_list)

    # make links clickable
    pwd = Path().resolve()

    predictor_values = []
    dmr_values = []
    wdmr_values = []
    for _, row in output_df.iterrows():
        p = row["Predictor"]
        print(p)
        predictor_values.append(
            "<a href='/{}'>{}</a>".format(
                f"/{pwd}/output/{p}-feature-plot.html", row["Predictor"]
            )
        )
        dmr_values.append(
            "<a href='/{}'>{}</a>".format(
                f"/{pwd}/output/{p}-DMR-plot.html", row["DMR"]
            )
        )
        wdmr_values.append(
            "<a href='/{}'>{}</a>".format(
                f"/{pwd}/output/{p}-wDMR-table.html", row["wDMR"]
            )
        )

    output_df["Predictor"] = predictor_values
    output_df["DMR"] = dmr_values
    output_df["wDMR"] = wdmr_values

    with open("./output/feature-comparison-table.html", "w") as out:
        out.write(
            output_df.to_html(
                index=False,
                na_rep="None",
                render_links=True,
                escape=False,
            )
        )


if __name__ == "__main__":
    sys.exit(main())
