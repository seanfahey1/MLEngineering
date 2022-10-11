import random
import sys

import pandas as pd
import plotly.express as px
import statsmodels.api

debug = True


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


def inspect_cont_pred_cont_resp(df, predictor, response):
    """
    Generates scatter plot with linear regression line for CONTINUOUS predictor and CONTINUOUS response.
    """
    if debug:
        return
    p_value, t_value = do_regression(df, predictor, response, method="linear")

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
    fig.show()


def inspect_bool_pred_cont_resp(df, predictor, response, reverse=False):
    """
    Generates violin plot grouped by the predictor when the predictor is CATEGORICAL/BOOLEAN and the response is
    CONTINUOUS. This function also works for the reverse (switching predictor and response) when reverse is set to
    True.
    """
    if debug:
        pass

    if reverse:
        cat = response
        cont = predictor
        # p_value, t_value = do_regression(df, predictor, response, method="logistic-reversed")
        title_text = "continuous v. categorical"
    else:
        cat = predictor
        cont = response
        # p_value, t_value = do_regression(df, predictor, response, method="logistic")
        title_text = "categorical v. continuous"

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
    fig.show()


def inspect_bool_pred_bool_resp(df, predictor, response):
    """
    Generates a heat map showing the frequency of each response value relative to each predictor value when both the
    predictor and response are CATEGORICAL/BOOLEAN.
    """
    # TODO: should this be normalized by the predictor?? Probably?
    if debug:
        return
    count_df = (
        df[[predictor, response]]
        .value_counts()
        .reset_index()
        .pivot(index=predictor, columns=response, values=0)
    )
    print(count_df)
    fig = px.imshow(count_df)
    fig.update_layout(
        title=f"<b>{predictor} v. {response}</b><br>continuous v. continuous",
        xaxis_title=response,
        yaxis_title=predictor,
    )
    fig.show()


def do_regression(df, predictor, response, method):
    # drop any NaN values
    print(predictor, response)
    df = df[[predictor, response]].dropna()
    print(df)

    # do p and t calculations
    if method == "linear":
        pred = statsmodels.api.add_constant(df[predictor])
        fitted_output = statsmodels.api.OLS(df[response], pred).fit()

    # elif method == "logistic":
    #     pred = statsmodels.api.add_constant(df[predictor])
    #     fitted_output = statsmodels.api.Logit(df[predictor], df[response]).fit()
    #
    # elif method == "logistic-reversed":
    #     print('reversed')
    #     print(df[predictor])
    #     pred = statsmodels.api.add_constant(df[response].to_numpy())
    #     fitted_output = statsmodels.api.Logit(df[predictor].to_numpy(), pred).fit()

    else:
        raise ValueError("Invalid method")

    # Get the stats!
    t_value = round(fitted_output.tvalues[1], 6)
    p_value = "{:.6e}".format(fitted_output.pvalues[1])
    return p_value, t_value


def dataset_insert():
    """
    Function to easily drop in different csv formatted datasets and lists of predictor and response columns to use.
    """
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jorisvandenbossche/pandas-tutorial/master/data/titanic.csv"
    )
    df["test1"] = [random.choice([True, False]) for i in range(len(df))]
    df["test2"] = [random.choice([0, 1]) for i in range(len(df))]
    predictors = ["Sex", "Age", "SibSp", "Parch", "Fare", "Pclass", "test1", "test2"]
    response = "Survived"
    # response = "Fare"
    return df, predictors, response


def main():
    # Using titanic dataset as placeholder for testing. Any pandas dataframe can be inserted here
    df, predictors, response = dataset_insert()

    response_continuous = determine_continuous(df, response)
    cont_lookup = dict()

    # loop through each predictor and generate the correct plot
    for predictor in predictors:
        # determine if predictor is categorical or continuous
        predictor_continuous = determine_continuous(df, predictor)
        cont_lookup[predictor] = predictor_continuous

        # use cat/cont info to determine which plotting function to use
        if response_continuous and predictor_continuous:
            inspect_cont_pred_cont_resp(df, predictor, response)
        elif response_continuous and not predictor_continuous:
            inspect_bool_pred_cont_resp(df, predictor, response)
        elif not response_continuous and predictor_continuous:
            inspect_bool_pred_cont_resp(df, predictor, response, reverse=True)
        elif not response_continuous and not predictor_continuous:
            inspect_bool_pred_bool_resp(df, predictor, response)


if __name__ == "__main__":
    sys.exit(main())
