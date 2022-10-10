import sys

import pandas as pd
import plotly.graph_objects as go


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
    fig = go.Fgure(
        go.Scatter(
            x=df[predictor],
            y=df[response],
        )
    )
    fig.show()


def main():
    # Using titanic dataset as placeholder for testing. Any pandas dataframe can be inserted here
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jorisvandenbossche/pandas-tutorial/master/data/titanic.csv"
    )
    predictors = ["Sex", "Age", "SibSp", "Parch", "Fare", "Pclass", "Embarked"]
    response = "Survived"
    response_continuous = determine_continuous(df, response)

    for predictor in predictors:
        continuous = determine_continuous(df, predictor)
        print(response_continuous, continuous)


if __name__ == "__main__":
    sys.exit(main())
