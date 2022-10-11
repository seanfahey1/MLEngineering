import sys

import pandas as pd
import plotly.express as px

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
    fig = px.scatter(
        x=df[predictor],
        y=df[response],
        trendline="ols",
        title=f"<b>{predictor} v. {response}<b/><br>continuous v. continuous",
    )
    fig.update_layout(
        xaxis_title=predictor,
        yaxis_title=response,
    )
    fig.show()


def inspect_bool_pred_cont_resp(df, predictor, response):
    """
    Generates violin plot grouped by the predictor when the predictor is CATEGORICAL/BOOLEAN and the response is
    CONTINUOUS.
    """
    if debug:
        return
    fig = px.violin(
        color=df[predictor],
        y=df[response],
        points="all",
        box=True,
        title=f"<b>{predictor} v. {response}</b><br>categorical v. continuous",
    )
    fig.update_layout(
        xaxis_title=predictor,
        yaxis_title=response,
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


def main():
    # Using titanic dataset as placeholder for testing. Any pandas dataframe can be inserted here
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jorisvandenbossche/pandas-tutorial/master/data/titanic.csv"
    )
    predictors = ["Sex", "Age", "SibSp", "Parch", "Fare", "Pclass", "Embarked"]
    response = "Survived"
    # response = "Age"
    response_continuous = determine_continuous(df, response)

    for predictor in predictors:
        predictor_continuous = determine_continuous(df, predictor)
        if response_continuous and predictor_continuous:
            inspect_cont_pred_cont_resp(df, predictor, response)
        elif response_continuous and not predictor_continuous:
            inspect_bool_pred_cont_resp(df, predictor, response)
        elif not response_continuous and not predictor_continuous:
            inspect_bool_pred_bool_resp(df, predictor, response)


if __name__ == "__main__":
    sys.exit(main())
