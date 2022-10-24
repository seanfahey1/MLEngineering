#!/usr/bin/env python3
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api
from cat_correlation import cat_cont_correlation_ratio, cat_correlation
from plotly.io import to_html
from scipy import stats


def cat_cat_correlation_metrics(df, col1, col2):
    pass


def cat_cont_correlation_metrics(df, col1, col2):
    pass


def cont_cont_correlation_metrics(df, col1, col2):
    pass


def difference_w_mean_of_resp_1d(df, predictor, response):
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
    df = df[[predictor, response]].dropna()
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
        previous_bin_max = bin_max

    msd_value = np.mean(msd_df["MeanSquaredDiff"])
    msd_weighted_value = np.mean(msd_df["MeanSquaredDiffWeighted"])

    msd_plot = plot_msd(msd_df, predictor)

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


def brute_force_cat_cat():
    pass


def brute_force_cat_cont():
    pass


def brute_force_cont_cont():
    pass


def dmr_2d_cat_cat(df, catpred1, catpred2, response, output_dir):
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

    previous_bin_max = min(df[contpred])  # TODO: is this the soruce of the error?

    # bin by categorical first
    for row, val in enumerate(row_labels):
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


def determine_cat_cont(df, predictors, response):
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

        cont_predictors.append(predictor)

    try:
        _ = [float(x) for x in df[response].unique()]
        response_cat = False
    except ValueError:
        response_cat = True

    return cat_predictors, cont_predictors, response_cat


def dataset_insert_data():
    """
    Function to easily drop in different csv formatted datasets and lists of predictor and response columns to use.
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
    return df, predictors, response


def linear_regression(df, predictor, response):
    pred = statsmodels.api.add_constant(df[predictor])
    fitted_output = statsmodels.api.OLS(df[response], pred).fit(disp=0)
    t_value = round(fitted_output.tvalues[1], 6)
    p_value = "{:.6e}".format(fitted_output.pvalues[1])
    res = stats.linregress(df[predictor], df[response])

    return p_value, t_value, res.rvalue


def main():
    # setup output directory
    cwd = pathlib.Path(__file__).parent.resolve()
    output_dir = cwd / "output"
    pathlib.Path(output_dir).mkdir(exist_ok=True)

    # setup empty html files to build onto
    html_predictor_comparison_table_output_file = (
        f"{output_dir}/predictor-comparison-tables.html"
    )
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
            "absolute value of corr ratio",
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
            abs(corr),
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

    cat_cat_correlation_plot = px.imshow(cat_cat_correlation_array, text_auto=True)
    cat_cat_correlation_plot.update_layout(
        title="Cramer's V correlation metric between predictors",
        xaxis_title="categorical predictors",
        yaxis_title="categorical predictors",
    )

    # write html table outputs
    with open(html_predictor_comparison_table_output_file, "a") as out:
        out.write("<h1>Categorical-Categorical Predictor Comparisons</h1>")
        out.write("<h3>Weighted Difference of Mean Response")
        out.write(
            brute_force_table_cat_cat.to_html(
                index=False,
                na_rep="None",
                render_links=True,
                escape=False,
            )
        )
        out.write("<br>Correlation Heat Plots")
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
            "correlation ratio (Pearson's)",
            "absolute value of corr ratio",
            "link to heat plot",
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
                abs(corr),
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
        by="correlation ratio (Pearson's)", ascending=False, inplace=True
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
        out.write("<h3>Weighted Difference of Mean Response")
        out.write(
            brute_force_table_cat_cont.to_html(
                index=False,
                na_rep="None",
                render_links=True,
                escape=False,
            )
        )
        out.write("<br>Correlation Violin Plots")
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
            "correlation ratio (R)",
            "R absolute value",
            "p-value",
            "t-value",
            "link to heat plot",
        ]
    )
    cont_cont_correlation_array = pd.DataFrame(
        columns=cont_predictors, index=cont_predictors
    )
    for contpred1, contpred2 in itertools.combinations(cont_predictors, 2):
        # continuous v. continuous correlation ratio
        p_value, t_value, r_value = linear_regression(df, contpred1, contpred2)

        cont_cont_correlation_array[contpred1][contpred2] = abs(r_value)
        cont_cont_correlation_array[contpred1][contpred2] = abs(r_value)

        plot_cont_cont(df, contpred1, contpred2, output_dir)

        correlation_table_cont_cont.loc[len(correlation_table_cont_cont)] = [
            contpred1,
            contpred2,
            "continuous",
            "continuous",
            r_value,
            abs(r_value),
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
        by="R absolute value", ascending=False, inplace=True
    )
    cont_cont_correlation_plot = px.imshow(cont_cont_correlation_array, text_auto=True)
    cont_cont_correlation_plot.update_layout(
        title="R correlation metric between predictors",
        xaxis_title="continuous predictors",
        yaxis_title="continuous predictors",
    )

    # write html table outputs
    with open(html_predictor_comparison_table_output_file, "a") as out:
        out.write("<br><br><h1>Continuous-Continuous Predictor Comparisons</h1>")
        out.write("<h3>Weighted Difference of Mean Response")
        out.write(
            brute_force_table_cont_cont.to_html(
                index=False,
                na_rep="None",
                render_links=True,
                escape=False,
            )
        )
        out.write("<br>Correlation Scatter Plots")
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


if __name__ == "__main__":
    sys.exit(main())
