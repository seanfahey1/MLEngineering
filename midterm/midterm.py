#!/usr/bin/env python3
import itertools
import pathlib
import sys
from distutils import util

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api
from cat_correlation import cat_cont_correlation_ratio, cat_correlation
from dataset_loader import get_test_data_set  # noqa: F401
from plotly.io import to_html
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def test_cat_response(values):
    """
    Test a categorical response column if it can be converted to a bool/float/int
    @param values: single DF column
    @return: the same column, a list, or None. Plus a modifier to indicate if the input was modified or failed.
    """
    try:
        for x in values:
            _ = float(x)
        return values, False
    except ValueError:
        response_values = []
        try:
            for x in values:
                response_values.append(int(util.strtobool(x)))
            values = response_values, True
            return values
        except ValueError:
            return values, None


def difference_w_mean_of_resp_1d_cont_pred(df, predictor, response, out_dir):
    """
    Bins the predictor into 10 bins, then calculates the mean response value for each bin and the population mean
    response value.
    @param df: input dataframe with predictor and response columns
    @param predictor: a single predictor column name
    @param response: a single response column name
    @param out_dir: the pathlib Path to the output directory
    @return: difference with mean ot response df, dmr value, wdmr value
    """
    # returns DMR, wDMR, plot html
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
    df = df[[predictor, response]].dropna().reset_index()
    df = df.sort_values(by=predictor, ascending=True).reset_index(drop=True)

    # figure out number of samples to use per bin
    bin_step_sizes = [((df[predictor].max() - df[predictor].min()) / 10)] * 9 + [np.inf]
    # calculate mean response per bin, bin predictor min, bin predictor max
    previous_bin_max = min(df[predictor])
    pop_mean_response = np.mean(df[response])

    for i, bin_step_size in enumerate(bin_step_sizes):
        # get just the values for the current bin
        bin_df = df[
            (df[predictor] >= previous_bin_max)
            & (df[predictor] < previous_bin_max + bin_step_size)
        ]
        bin_count = len(bin_df)
        pop_porp = bin_count / len(df)
        # assign values to bin
        if bin_count > 0:
            bin_min = min(bin_df[predictor])
            bin_max = max(bin_df[predictor])
            bin_center = (previous_bin_max + bin_step_size) / 2 + previous_bin_max
            mean_response = np.mean(bin_df[response])
            msd = (mean_response - pop_mean_response) ** 2 / bin_count
            msd_weighted = msd * pop_porp

        else:
            bin_min = previous_bin_max
            bin_max = previous_bin_max + bin_step_size
            bin_center = (previous_bin_max + bin_step_size) / 2 + previous_bin_max
            mean_response = None
            msd = None
            msd_weighted = None

        # assign values to the next row in the df
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
        previous_bin_max += bin_step_size

    msd_value = np.mean(msd_df["MeanSquaredDiff"])
    msd_weighted_value = np.mean(msd_df["MeanSquaredDiffWeighted"])

    msd_plot = plot_msd(msd_df, predictor)

    with open(
        out_dir / f"msd plot {predictor} v. response ({response}).html", "w"
    ) as outfile:
        outfile.write(msd_plot)

    return msd_df, msd_value, msd_weighted_value


def difference_w_mean_of_resp_1d_cat_pred(df, predictor, response, out_dir):
    """
    Bins the predictor into unique value bins, then calculates the mean response value for each bin and the population
    mean response value.
    @param df: input dataframe with predictor and response columns
    @param predictor: a single predictor column name
    @param response: a single response column name
    @param out_dir: the pathlib Path to the output directory
    @return: difference with mean ot response df, dmr value, wdmr value
    """
    # returns DMR, wDMR, plot html
    msd_df = pd.DataFrame(
        columns=[
            "(𝑖)",
            "BinCenters",
            "BinCount",
            "BinMeans (𝜇𝑖)",
            "PopulationMean (𝜇𝑝𝑜𝑝)",
            "MeanSquaredDiff",
            "PopulationProportion (𝑤𝑖)",
            "MeanSquaredDiffWeighted",
        ]
    )
    df = df[[predictor, response]].dropna().reset_index(drop=True)
    df = df.sort_values(by=predictor, ascending=True).reset_index(drop=True)

    pop_mean_response = np.mean(df[response])

    for i, pred_value in enumerate(df[predictor].unique()):
        # get just the values for the current bin
        bin_df = df[df[predictor] == pred_value]
        bin_count = len(bin_df)
        pop_porp = bin_count / len(df)
        bin_center = pred_value
        # assign values to bin
        if bin_count > 0:
            mean_response = np.mean(bin_df[response])
            msd = (mean_response - pop_mean_response) ** 2 / bin_count
            msd_weighted = msd * pop_porp

        else:
            mean_response = None
            msd = None
            msd_weighted = None

        # assign values to the next row in the df
        msd_df.loc[len(msd_df)] = [
            str(int(i)),
            bin_center,
            bin_count,
            mean_response,
            pop_mean_response,
            msd,
            pop_porp,
            msd_weighted,
        ]

    msd_value = np.mean(msd_df["MeanSquaredDiff"])
    msd_weighted_value = np.mean(msd_df["MeanSquaredDiffWeighted"])

    msd_plot = plot_msd(msd_df, predictor)

    with open(
        out_dir / f"msd plot {predictor} v. response ({response}).html", "w"
    ) as outfile:
        outfile.write(msd_plot)

    return msd_df, msd_value, msd_weighted_value


def plot_msd(df, predictor):
    """
    Produces a plotly plot of the binned difference with mean of response
    @param df: A pandas DF with the necessary difference w/ mean response values.
    @param predictor: The name of the predictor (for plot title/legend)
    @return: html of a plotly plot for that predictor
    """
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
    fig.add_trace(
        go.Scatter(
            name="𝜇𝑝𝑜𝑝 - 𝜇𝑖",
            x=df["BinCenters"],
            y=df["BinMeans (𝜇𝑖)"],
            yaxis="y2",
        )
    )
    fig.update_layout(title=f"difference with mean of response - {predictor}")
    fig.update_layout(
        xaxis_title=f"{predictor} bin",
        yaxis_title="Population",
        yaxis2_title="Response",
    )
    fig.update_yaxes(rangemode="tozero")
    # fig.show()

    return to_html(fig, include_plotlyjs="cdn")


def dmr_2d_cat_cat(df, catpred1, catpred2, response, output_dir):
    """
    Produces a 2-D DMR plot between two categorical features. Each bin is each unique value for each of the two
    features.
    @param df: The input dataframe with predictor columns and a response column
    @param catpred1: The name of predictor column #1
    @param catpred2: The name of predictor column #2
    @param response: The name of the response column
    @param output_dir: a pathlib Path to the output directory
    @return: average DMR and weighted average DMR for the pair of predictors
    """
    # test if response can be treated as numeric (ie. boolean or 0/1 int type values)
    response_values, modified = test_cat_response(df[response])
    if modified is True:
        df[response] = response_values
    elif modified is None:
        # can't work with this data :(
        with open(
            output_dir
            / f"predictor v. predictor average response - {catpred1} v. {catpred2}.html",
            "w",
        ) as out:
            out.write("")
        with open(
            output_dir
            / f"predictor v. predictor population proportion - {catpred1} v. {catpred2}.html",
            "w",
        ) as out:
            out.write("")
        return None, None

    # setup empty arrays to plot
    avg_resp_array = np.zeros((len(df[catpred1].unique()), len(df[catpred2].unique())))
    pop_size_array = np.zeros((len(df[catpred1].unique()), len(df[catpred2].unique())))
    row_labels = df[catpred1].unique()
    col_labels = df[catpred2].unique()
    dmr_list = []
    wdmr_list = []

    # bin by categorical for both
    pop_mean_resp = np.mean(df[response])
    for row, val1 in enumerate(row_labels):
        for col, val2 in enumerate(col_labels):

            bin = df[(df[catpred1] == val1) & (df[catpred2] == val2)]
            bin_size = len(bin)

            if bin_size > 0:
                bin_mean_resp = np.mean(bin[response])
                dmr = (bin_mean_resp - pop_mean_resp) ** 2 / bin_size
                pop_porp = bin_size / len(df)
                wdmr = dmr * pop_porp
            else:
                pop_porp = 0
                bin_mean_resp = None
                dmr = None
                wdmr = None
            dmr_list.append(dmr)
            wdmr_list.append(wdmr)
            avg_resp_array[row][col] = bin_mean_resp
            pop_size_array[row][col] = pop_porp

    avg_dmr = np.mean([x for x in dmr_list if x is not None])
    avg_wdmr = np.mean([x for x in wdmr_list if x is not None])

    pop_size_fig = px.imshow(
        pop_size_array,
        text_auto=True,
        labels=dict(x=catpred2, y=catpred1),
        x=col_labels,
        y=row_labels,
        title=f"predictor v. predictor population proportion - {catpred1} v. {catpred2}",
        aspect="auto",
    )
    pop_size_html = pop_size_fig.to_html()

    with open(
        output_dir
        / f"predictor v. predictor population proportion - {catpred1} v. {catpred2}.html",
        "w",
    ) as out:
        out.write(pop_size_html)

    avg_resp_fig = px.imshow(
        avg_resp_array,
        text_auto=True,
        labels=dict(x=catpred2, y=catpred1),
        x=col_labels,
        y=row_labels,
        title=f"predictor v. predictor average response - {catpred1} v. {catpred2}",
        aspect="auto",
    )
    avg_resp_html = avg_resp_fig.to_html()

    with open(
        output_dir
        / f"predictor v. predictor average response - {catpred1} v. {catpred2}.html",
        "w",
    ) as out:
        out.write(avg_resp_html)

    # avg_resp_fig.show()
    # pop_size_fig.show()
    return avg_dmr, avg_wdmr


def dmr_2d_cat_cont(df, catpred, contpred, response, output_dir):
    """
    Produces a 2-D DMR plot between a categorical and a continuous feature. Each bin is each unique value for each
    of the two features.
    @param df: The input dataframe with predictor columns and a response column
    @param catpred: The name of the categorical predictor column
    @param contpred: The name of the continuous predictor column
    @param response: The name of the response column
    @param output_dir: a pathlib Path to the output directory
    @return: average DMR and weighted average DMR for the pair of predictors
    """
    # setup empty arrays to plot
    avg_resp_array = np.zeros((len(df[catpred].unique()), 10))
    pop_size_array = np.zeros((len(df[catpred].unique()), 10))

    # get continuous column bin sizes
    df = df.sort_values(by=contpred, ascending=True).reset_index(drop=True)
    bin_step_sizes = [((df[contpred].max() - df[contpred].min()) / 10)] * 9 + [np.inf]

    pop_mean_resp = np.mean(df[response])
    row_labels = df[catpred].unique()
    col_labels = [
        (min(df[contpred]) + bin_step_sizes[0] * x) / 2 + min(df[contpred])
        for x in range(1, 11)
    ]

    dmr_list = []
    wdmr_list = []

    # bin by categorical first
    for row, val in enumerate(row_labels):
        previous_bin_max = min(df[contpred])
        for col, bin_step_size in enumerate(bin_step_sizes):
            bin = df[
                (df[contpred] >= previous_bin_max)
                & (df[contpred] < previous_bin_max + bin_step_size)
            ]
            bin = bin[bin[catpred] == val]
            bin_size = len(bin)

            if bin_size > 0:
                bin_mean_resp = np.mean(bin[response])
                dmr = (bin_mean_resp - pop_mean_resp) ** 2 / bin_size
                pop_porp = bin_size / len(df)
                wdmr = dmr * pop_porp
            else:
                pop_porp = 0
                bin_mean_resp = None
                dmr = None
                wdmr = None
            dmr_list.append(dmr)
            wdmr_list.append(wdmr)
            avg_resp_array[row][col] = bin_mean_resp
            pop_size_array[row][col] = pop_porp
            previous_bin_max = previous_bin_max + bin_step_size

    avg_dmr = np.mean([x for x in dmr_list if x is not None])
    avg_wdmr = np.mean([x for x in wdmr_list if x is not None])
    pop_size_fig = px.imshow(
        pop_size_array,
        text_auto=True,
        labels=dict(x=contpred, y=catpred),
        x=col_labels,
        y=row_labels,
        title=f"predictor v. predictor population proportion - {catpred} v. {contpred}",
        aspect="auto",
    )
    pop_size_html = pop_size_fig.to_html()

    with open(
        output_dir
        / f"predictor v. predictor population proportion - {catpred} v. {contpred}.html",
        "w",
    ) as out:
        out.write(pop_size_html)

    avg_resp_fig = px.imshow(
        avg_resp_array,
        text_auto=True,
        labels=dict(x=contpred, y=catpred),
        x=col_labels,
        y=row_labels,
        title=f"predictor v. predictor average response - {catpred} v. {contpred}",
        aspect="auto",
    )
    avg_resp_html = avg_resp_fig.to_html()

    with open(
        output_dir
        / f"predictor v. predictor average response - {catpred} v. {contpred}.html",
        "w",
    ) as out:
        out.write(avg_resp_html)

    # avg_resp_fig.show()
    # pop_size_fig.show()
    return avg_dmr, avg_wdmr


def dmr_2d_cont_cont(df, contpred1, contpred2, response, output_dir):
    """
    Produces a 2-D DMR plot between two continuous features. Each bin is each unique value for each of the two
    features.
    @param df: The input dataframe with predictor columns and a response column
    @param contpred1: The name of predictor column #1
    @param contpred2: The name of predictor column #2
    @param response: The name of the response column
    @param output_dir: a pathlib Path to the output directory
    @return: average DMR and weighted average DMR for the pair of predictors
    """
    # setup empty arrays to plot
    avg_resp_array = np.zeros((10, 10))
    pop_size_array = np.zeros((10, 10))

    # get continuous column bin sizes for predictor 1
    df = df.sort_values(by=contpred1, ascending=True).reset_index(drop=True)
    pred_1_bin_step_sizes = [((df[contpred1].max() - df[contpred1].min()) / 10)] * 9 + [
        np.inf
    ]

    # get continuous column bin sizes for predictor 2
    df = df.sort_values(by=contpred2, ascending=True).reset_index(drop=True)
    pred_2_bin_step_sizes = [((df[contpred2].max() - df[contpred2].min()) / 10)] * 9 + [
        np.inf
    ]

    pop_mean_resp = np.mean(df[response])
    row_labels = [
        (min(df[contpred1]) + pred_1_bin_step_sizes[0] * x) / 2 + min(df[contpred1])
        for x in range(1, 11)
    ]
    col_labels = [
        (min(df[contpred2]) + pred_2_bin_step_sizes[0] * x) / 2 + min(df[contpred2])
        for x in range(1, 11)
    ]

    dmr_list = []
    wdmr_list = []

    pred_1_previous_bin_max = min(df[contpred1])
    for row, pred_1_bin_step_size in enumerate(pred_1_bin_step_sizes):
        pred_2_previous_bin_max = min(df[contpred2])
        for col, pred_2_bin_step_size in enumerate(pred_2_bin_step_sizes):
            bin = df[
                (df[contpred1] >= pred_1_previous_bin_max)
                & (df[contpred1] < pred_1_previous_bin_max + pred_1_bin_step_size)
            ]

            bin = bin[
                (bin[contpred2] >= pred_2_previous_bin_max)
                & (bin[contpred2] < pred_2_previous_bin_max + pred_2_bin_step_size)
            ]

            bin_size = len(bin)

            if bin_size > 0:
                bin_mean_resp = np.mean(bin[response])
                dmr = (bin_mean_resp - pop_mean_resp) ** 2 / bin_size
                pop_porp = bin_size / len(df)
                wdmr = dmr * pop_porp
            else:
                pop_porp = 0
                bin_mean_resp = None
                dmr = None
                wdmr = None
            dmr_list.append(dmr)
            wdmr_list.append(wdmr)
            avg_resp_array[row][col] = bin_mean_resp
            pop_size_array[row][col] = pop_porp
            pred_2_previous_bin_max = pred_2_previous_bin_max + pred_2_bin_step_size
        pred_1_previous_bin_max = pred_1_previous_bin_max + pred_1_bin_step_size

    avg_dmr = np.mean([x for x in dmr_list if x is not None])
    avg_wdmr = np.mean([x for x in wdmr_list if x is not None])

    pop_size_fig = px.imshow(
        pop_size_array,
        text_auto=True,
        labels=dict(x=contpred1, y=contpred2),
        x=col_labels,
        y=row_labels,
        title=f"predictor v. predictor population proportion - {contpred1} v. {contpred2}",
        aspect="auto",
    )
    pop_size_html = pop_size_fig.to_html()

    with open(
        output_dir
        / f"predictor v. predictor population proportion - {contpred1} v. {contpred2}.html",
        "w",
    ) as out:
        out.write(pop_size_html)

    avg_resp_fig = px.imshow(
        avg_resp_array,
        text_auto=True,
        labels=dict(x=contpred1, y=contpred2),
        x=col_labels,
        y=row_labels,
        title=f"predictor v. predictor average response - {contpred1} v. {contpred2}",
        aspect="auto",
    )
    avg_resp_html = avg_resp_fig.to_html()

    with open(
        output_dir
        / f"predictor v. predictor average response - {contpred1} v. {contpred2}.html",
        "w",
    ) as out:
        out.write(avg_resp_html)

    # avg_resp_fig.show()
    # pop_size_fig.show()
    return avg_dmr, avg_wdmr


def plot_cat_cat(df, column1, column2, output_dir):
    """
    Plots correlation between two categorical predictors
    @param df: The input dataframe with predictor columns
    @param column1: The first categorical predictor column name
    @param column2: The second categorical predictor column name
    @param output_dir: Pathlib Path to the output directory
    """
    count_df = (
        df[[column1, column2]]
        .value_counts()
        .reset_index()
        .pivot(index=column1, columns=column2, values=0)
    )
    fig = px.imshow(count_df)
    fig.update_layout(
        title=f"<b>{column1} v. {column2}</b><br>categorical v. categorical<br>",
        xaxis_title=column1,
        yaxis_title=column2,
    )
    with open(
        output_dir
        / f"predictor v. predictor correlation plot - {column1} v. {column2}.html",
        "w",
    ) as out:
        out.write(to_html(fig, include_plotlyjs="cdn"))


def plot_cat_cont(df, cat_col, cont_col, output_dir):
    """
    Plots correlation between a categorical and continuous predictor
    @param df: The input dataframe with predictor columns
    @param cat_col: The categorical predictor column name
    @param cont_col: The continuous predictor column name
    @param output_dir: Pathlib Path to the output directory
    """
    fig = px.violin(
        color=df[cat_col],
        y=df[cont_col],
        points="all",
        box=True,
    )
    fig.update_layout(
        title=f"<b>{cat_col} v. {cont_col}</b><br>categorical v. continuous",
        xaxis_title=cat_col,
        yaxis_title=cont_col,
    )
    with open(
        output_dir
        / f"predictor v. predictor correlation plot - {cat_col} v. {cont_col}.html",
        "w",
    ) as out:
        out.write(to_html(fig, include_plotlyjs="cdn"))


def plot_cont_cont(df, column1, column2, output_dir):
    """
    Plots correlation between two continuous predictors
    @param df: The input dataframe with predictor columns
    @param column1: The first continuous predictor column name
    @param column2: The second continuous predictor column name
    @param output_dir: Pathlib Path to the output directory
    """
    fig = px.scatter(
        x=df[column1],
        y=df[column2],
        trendline="ols",
    )
    fig.update_layout(
        title=f"<b>{column1} v. {column2}</b><br>continuous v. continuous<br>",
        xaxis_title=column1,
        yaxis_title=column2,
    )
    with open(
        output_dir
        / f"predictor v. predictor correlation plot - {column1} v. {column2}.html",
        "w",
    ) as out:
        out.write(to_html(fig, include_plotlyjs="cdn"))


def plot_cont_pred_cont_resp(df, predictor, response, out_dir):
    """
    Plots correlation between a continuous predictor and a continuous response
    @param df: The input dataframe with predictor columns
    @param predictor: The continuous predictor column name
    @param response: The continuous response column name
    @param out_dir: Pathlib Path to the output directory
    """
    fig = px.scatter(
        x=df[predictor],
        y=df[response],
        trendline="ols",
        title=f"<b>{predictor} v. {response}</b><br>continuous v. continuous<br>",
    )
    fig.update_layout(
        xaxis_title=predictor,
        yaxis_title=response,
    )
    # fig.show()

    with open(out_dir / f"{predictor} v. response.html", "w") as out:
        out.write(to_html(fig, include_plotlyjs="cdn"))

    return


def plot_cat_pred_cont_resp(df, predictor, response, out_dir):
    """
    Plots correlation between a categorical predictor and a continuous response
    @param df: The input dataframe with predictor columns
    @param predictor: The categorical predictor column name
    @param response: The continuous response column name
    @param out_dir: Pathlib Path to the output directory
    """
    fig = px.violin(
        color=df[predictor],
        y=df[response],
        points="all",
        box=True,
        title=f"<b>{predictor} v. {response}</b>",
    )
    fig.update_layout(
        xaxis_title=predictor,
        yaxis_title=response,
    )
    # fig.show()

    with open(out_dir / f"{predictor} v. response.html", "w") as out:
        out.write(to_html(fig, include_plotlyjs="cdn"))

    return


def plot_cont_pred_cat_resp(df, predictor, response, out_dir):
    """
    Plots correlation between a continuous predictor and a categorical response
    @param df: The input dataframe with predictor columns
    @param predictor: The continuous predictor column name
    @param response: The categorical response column name
    @param out_dir: Pathlib Path to the output directory
    """
    fig = px.violin(
        color=df[response],
        y=df[predictor],
        points="all",
        box=True,
        title=f"<b>{response} v. {predictor}</b>",
    )
    fig.update_layout(
        xaxis_title=response,
        yaxis_title=predictor,
    )
    # fig.show()

    with open(out_dir / f"{predictor} v. response.html", "w") as out:
        out.write(to_html(fig, include_plotlyjs="cdn"))

    return


def plot_cat_pred_cat_resp(df, predictor, response, out_dir):
    """
    Plots correlation between a categorical predictor and a categorical response
    @param df: The input dataframe with predictor columns
    @param predictor: The categorical predictor column name
    @param response: The categorical response column name
    @param out_dir: Pathlib Path to the output directory
    """
    count_df = (
        df[[predictor, response]]
        .value_counts()
        .reset_index()
        .pivot(index=predictor, columns=response, values=0)
    )
    fig = px.imshow(count_df)
    fig.update_layout(
        title=f"<b>{predictor} v. {response}</b>",
        xaxis_title=response,
        yaxis_title=predictor,
    )
    # fig.show()

    with open(out_dir / f"{predictor} v. response.html", "w") as out:
        out.write(to_html(fig, include_plotlyjs="cdn"))

    return


def determine_cat_cont(df, predictors, response):
    """
    Determine which predictors in a dataframe are continuous and which are categorical. Also determines if response
    to be predicted is categorical.
    @param df: The input dataframe with predictor columns and response column
    @param predictors: A list of all predictor column names
    @param response: The response column name
    @return: list of cat predictors, list of cont predictors, a boolean indicating if the response is categorical
    """

    class FoundCategorical(Exception):
        pass

    cat_predictors, cont_predictors = [], []
    for predictor in predictors:
        col_values = df[predictor].unique()
        try:
            for value in col_values:
                try:
                    float(value)
                except ValueError:
                    cat_predictors.append(predictor)
                    raise FoundCategorical
        except FoundCategorical:
            continue
        if len(df[predictor].unique()) <= 2:
            cat_predictors.append(predictor)
        else:
            cont_predictors.append(predictor)

    try:
        _ = [float(x) for x in df[response].unique()]
        response_cat = False
    except ValueError:
        response_cat = True

    # handle an exception for boolean true/false values that can be converted to floats
    if len(df[response].unique()) == 2:
        response_cat = True

    return cat_predictors, cont_predictors, response_cat


def dataset_insert_data():
    """
    Function to easily drop in different csv formatted datasets and lists of predictor and response columns to use.
    @return: pandas DataFrame, list of predictor column names, response column name
    """
    df = pd.read_csv("/Users/sean/workspace/Sean/sdsu/BDA602/heart disease data.csv")

    # response = "age"
    response = "Heart Disease"
    predictors = [
        "age",
        "sex",
        "chest pain type",
        "resting blood pressure",
        "serum cholestrol mg per dL",
        "resting EKG value",
        "maximum heart rate",
        "ST depression",
        "flourosopy results",
        "thal",
    ]

    predictors = [x for x in predictors if x != response]
    df, predictors, response = get_test_data_set(data_set_name="titanic_2")
    df = df.reset_index(drop=True)

    return df, predictors, response


def linear_regression(df, predictor, response):
    """
    calculates p-value, t-value, and pearson's correlation coefficient between a predictor and a response
    @param df: The input dataframe with predictor columns and response column
    @param predictor: The name of a continuous predictor column
    @param response: The name of a continuous response column
    @return: p-value, t-value, pearson's correlation value
    """
    df = df.dropna(axis=0)
    pred = statsmodels.api.add_constant(df[predictor])
    fitted_output = statsmodels.api.OLS(df[response], pred).fit(disp=0)
    t_value = round(fitted_output.tvalues[1], 6)
    p_value = "{:.6e}".format(fitted_output.pvalues[1])
    res = stats.pearsonr(df[predictor], df[response]).statistic

    return p_value, t_value, res


def random_forest(df, response, predictors, response_type):
    if response_type == "categorical":
        rf = RandomForestClassifier(random_state=0)
    elif response_type == "continuous":
        rf = RandomForestRegressor(random_state=0)
    else:
        raise ValueError("invalid response type")

    relevant_columns = [response] + predictors
    df = df[relevant_columns].dropna().reset_index(drop=True)
    X = df[predictors].to_numpy()

    if X.shape[1] == 1:
        X = X.reshape(-1, 1)
    y = df[response].to_numpy().reshape(-1, 1).ravel()
    rf.fit(X, y)
    return list(rf.feature_importances_)


def main():
    # setup output directory
    cwd = pathlib.Path(__file__).parent.resolve()
    output_dir = cwd / "output"
    pathlib.Path(output_dir).mkdir(exist_ok=True)

    # PREDICTOR V. PREDICTOR CORRELATIONS
    # setup empty html files to build onto
    html_predictor_comparison_table_output_file = f"{output_dir}/comparison-tables.html"
    with open(html_predictor_comparison_table_output_file, "w") as out:
        out.write("")

    df, predictors, response = dataset_insert_data()
    cat_predictors, cont_predictors, response_cat = determine_cat_cont(
        df, predictors, response
    )

    # categorical v. categorical predictors
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
        columns=cat_predictors, index=cat_predictors
    )
    # get all pairwise unique combinations of categorical predictors
    for catpred1, catpred2 in itertools.combinations(cat_predictors, 2):
        # cat. v. cat. correlation ratio
        corr = cat_correlation(df[catpred1], df[catpred2])
        cat_cat_correlation_array[catpred1][catpred2] = abs(corr)
        cat_cat_correlation_array[catpred2][catpred1] = abs(corr)

        plot_cat_cat(df, catpred1, catpred2, output_dir)

        correlation_table_cat_cat.loc[len(correlation_table_cat_cat)] = [
            catpred1,
            catpred2,
            "categorical",
            "categorical",
            corr,
            f"<a href='//{output_dir}/predictor v. predictor correlation plot - {catpred1} v. {catpred2}.html'>correlation plot - {catpred1} v. {catpred2}</a>",  # noqa: E501
        ]
        # cat. v. cat. wDMR
        dmr, wdmr = dmr_2d_cat_cat(df, catpred1, catpred2, response, output_dir)
        brute_force_table_cat_cat.loc[len(brute_force_table_cat_cat)] = [
            catpred1,
            catpred2,
            "categorical",
            "categorical",
            dmr,
            wdmr,
            f"<a href='//{output_dir}/predictor v. predictor population proportion - {catpred1} v. {catpred2}.html'>population proportion - {catpred1} v. {catpred2}</a>",  # noqa: E501
            f"<a href='//{output_dir}/predictor v. predictor average response - {catpred1} v. {catpred2}.html'>average response - {catpred1} v. {catpred2}</a>",  # noqa: E501
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

    # categorical v. continuous predictors
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
        columns=cat_predictors, index=cont_predictors
    )
    # get all combinations between categorical and continuous predictors
    for catpred in cat_predictors:
        for contpred in cont_predictors:
            # cat. v. cont. correlation ratio
            corr = cat_cont_correlation_ratio(df[catpred], df[contpred])
            cat_cont_correlation_array[catpred][contpred] = abs(corr)

            plot_cat_cont(df, catpred, contpred, output_dir)

            correlation_table_cat_cont.loc[len(correlation_table_cat_cont)] = [
                catpred,
                contpred,
                "categorical",
                "continuous",
                corr,
                f"<a href='//{output_dir}/predictor v. predictor correlation plot - {catpred} v. {contpred}.html'>correlation plot - {catpred} v. {contpred}</a>",  # noqa: E501
            ]
            # cat. v. cont. wDMR
            dmr, wdmr = dmr_2d_cat_cont(df, catpred, contpred, response, output_dir)
            brute_force_table_cat_cont.loc[len(brute_force_table_cat_cont)] = [
                catpred,
                contpred,
                "categorical",
                "continuous",
                dmr,
                wdmr,
                f"<a href='//{output_dir}/predictor v. predictor population proportion - {catpred} v. {contpred}.html'>population proportion - {catpred} v. {contpred}</a>",  # noqa: E501
                f"<a href='//{output_dir}/predictor v. predictor average response - {catpred} v. {contpred}.html'>average response - {catpred} v. {contpred}</a>",  # noqa: E501
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

    # continuous v. continuous predictors
    # compare all continuous predictors to all continuous predictors
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
        columns=cont_predictors, index=cont_predictors
    )
    for contpred1, contpred2 in itertools.combinations(cont_predictors, 2):
        # continuous v. continuous correlation ratio
        p_value, t_value, corr = linear_regression(df, contpred1, contpred2)

        cont_cont_correlation_array[contpred1][contpred2] = abs(corr)
        cont_cont_correlation_array[contpred2][contpred1] = abs(corr)

        plot_cont_cont(df, contpred1, contpred2, output_dir)

        correlation_table_cont_cont.loc[len(correlation_table_cont_cont)] = [
            contpred1,
            contpred2,
            "continuous",
            "continuous",
            corr,
            abs(corr),
            p_value,
            t_value,
            f"<a href='//{output_dir}/predictor v. predictor correlation plot - {contpred1} v. {contpred2}.html'>correlation plot - {contpred1} v. {contpred2}</a>",  # noqa: E501
        ]
        # cont. v. cont. wDMR
        dmr, wdmr = dmr_2d_cont_cont(df, contpred1, contpred2, response, output_dir)
        brute_force_table_cont_cont.loc[len(brute_force_table_cont_cont)] = [
            contpred1,
            contpred2,
            "continuous",
            "continuous",
            dmr,
            wdmr,
            f"<a href='//{output_dir}/predictor v. predictor population proportion - {contpred1} v. {contpred2}.html'>population proportion - {contpred1} v. {contpred2}</a>",  # noqa: E501
            f"<a href='//{output_dir}/predictor v. predictor average response - {contpred1} v. {contpred2}.html'>average response - {contpred1} v. {contpred2}</a>",  # noqa: E501
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

    # PREDICTOR V. RESPONSE CORRELATIONS
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
    feature_importances = random_forest(
        df, response, cont_predictors, "categorical" if response_cat else "continuous"
    )

    for i, predictor in enumerate(cont_predictors):
        if response_cat:
            plot_cont_pred_cat_resp(df, predictor, response, output_dir)
            corr = cat_cont_correlation_ratio(df[response], df[predictor])
            corr_type = "Correlation Ratio"
        else:
            plot_cont_pred_cont_resp(df, predictor, response, output_dir)
            _, _, corr = linear_regression(df, predictor, response)
            corr_type = "Pearson's"

        msd_df, dmr, wdmr = difference_w_mean_of_resp_1d_cont_pred(
            df, predictor, response, output_dir
        )

        output_df.loc[len(output_df)] = [
            response,
            predictor,
            "categorical" if response_cat else "continuous",
            "continuous",
            min(df[predictor]),
            max(df[predictor]),
            np.median(df[predictor]),
            corr,
            corr_type,
            feature_importances[i],
            dmr,
            wdmr,
            f"<a href='//{output_dir}/msd plot {predictor} v. response ({response}).html'>DMR - {predictor}</a>",  # noqa: E501
            f"<a href='//{output_dir}/{predictor} v. response.html'>{predictor} v. response</a>",  # noqa: E501
        ]

    # Then add categorical predictors
    for i, predictor in enumerate(cat_predictors):
        if response_cat:
            plot_cat_pred_cat_resp(df, predictor, response, output_dir)
            corr = cat_correlation(df[predictor], df[response])
            corr_type = "Cramer's V"
        else:
            plot_cat_pred_cont_resp(df, predictor, response, output_dir)
            corr = cat_cont_correlation_ratio(df[predictor], df[response])
            corr_type = "Correlation Ratio"

        msd_df, dmr, wdmr = difference_w_mean_of_resp_1d_cat_pred(
            df, predictor, response, output_dir
        )

        output_df.loc[len(output_df)] = [
            response,
            predictor,
            "categorical" if response_cat else "continuous",
            "categorical",
            None,
            None,
            None,
            corr,
            corr_type,
            None,
            dmr,
            wdmr,
            f"<a href='//{output_dir}/msd plot {predictor} v. response ({response}).html'>DMR - {predictor}</a>",  # noqa: E501
            f"<a href='//{output_dir}/{predictor} v. response.html'>{predictor} v. response</a>",  # noqa: E501
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


if __name__ == "__main__":
    sys.exit(main())
