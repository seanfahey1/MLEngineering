#!/usr/bin/env python3
"""
This is a simple program that explores the Iris dataset by plotting various features. Predictions are then made using
3 different SKLearn models.
"""

import argparse

import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import SGDClassifier as sgd
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model",
        choices=["rf", "knn", "sgd", "all"],
        required=False,
        default="all",
        help="Optional flag to select which model to use. By default, 'all' models are used.",
    )
    parser.add_argument(
        "-np",
        "--noplot",
        action="store_const",
        const=False,
        default=True,
        dest="plot",
        help="Optional flag to disable feature plotting.",
    )

    args = parser.parse_args()
    return args.model, args.plot


def get_iris():
    df = pd.read_csv(
        "https://teaching.mrsharky.com/data/iris.data",
        names=[
            "sepal length in cm",
            "sepal width in cm",
            "petal length in cm",
            "petal width in cm",
            "class",
        ],
        header=None,
    )

    # overwriting 2 errors according to the iris.names file
    df.loc[34] = [4.9, 3.1, 1.5, 0.2, "Iris-setosa"]
    df.loc[37] = [4.9, 3.6, 1.4, 0.1, "Iris-setosa"]
    df = df.dropna()
    return df


def make_plots(df):
    box = px.box(
        df,
        y=[
            "sepal length in cm",
            "sepal width in cm",
            "petal length in cm",
            "petal width in cm",
        ],
        color="class",
        title="Boxplot of all Measurements",
    )
    box.show()

    violin = px.violin(
        df,
        y=[
            "sepal length in cm",
            "sepal width in cm",
            "petal length in cm",
            "petal width in cm",
        ],
        color="class",
        title="Violin of all Measurements",
        points="all",
    )
    violin.show()

    heatmap = px.density_heatmap(
        df,
        y="sepal length in cm",
        x="sepal width in cm",
        marginal_x="histogram",
        marginal_y="histogram",
        nbinsx=40,
        nbinsy=40,
        title="Density Heatmap of Sepal Length v. Sepal Width showing 3 distinct clusters",
    )
    heatmap.show()

    pie = px.pie(
        df,
        values=[1 for i in range(len(df))],
        names="class",
        title="Pie Chart showing % of population assigned to each class",
    )
    pie.show()

    contour = px.density_contour(
        df,
        y="petal length in cm",
        x="petal width in cm",
        marginal_x="histogram",
        marginal_y="histogram",
        nbinsx=40,
        nbinsy=40,
        color="class",
        title="Density Contour of Petal Length v. Petal Width showing 3 distinct clusters",
    )
    contour.show()


def print_predictions(predictions, probability, y_test):
    score = sum(
        [1 if predictions[i] == y_test[i] else 0 for i in range(len(predictions))]
    )
    print(f"# correct:\t{score}/{len(predictions)}")
    print(f"accuracy:\t{score/len(predictions)}")

    print("\tprediction\tprobability")
    for i in range(len(predictions)):
        print(f"\t{predictions[i]}\t{probability[i]}")

    print("\n\n")


def make_predictions(df, model):
    # setting up data
    X = df[
        [
            "sepal length in cm",
            "sepal width in cm",
            "petal length in cm",
            "petal width in cm",
        ]
    ].values
    y = df["class"].values

    # Using every 8th value as a test value instead of making fake data.
    X_test = X[::8]
    y_test = y[::8]

    # Setting up pipelines
    if model == "all" or model == "rf":
        # RF pipeline
        rf_pipeline = Pipeline(
            [
                ("StandardScaler", StandardScaler()),
                ("RFClassifier", rfc(random_state=123)),
            ]
        )
        rf_pipeline.fit(X, y)

        print("---RANDOM FOREST PREDICTION---")
        predictions, probability = rf_pipeline.predict(
            X_test
        ), rf_pipeline.predict_proba(X_test)
        print_predictions(predictions, probability, y_test)

    if model == "all" or model == "knn":
        # KNN pipeline
        knn_pipeline = Pipeline(
            [
                ("StandardScaler", StandardScaler()),
                ("KNNClassifier", knn(n_neighbors=5)),
            ]
        )
        knn_pipeline.fit(X, y)

        print("---KNN PREDICTION---")
        predictions, probability = knn_pipeline.predict(
            X_test
        ), knn_pipeline.predict_proba(X_test)
        print_predictions(predictions, probability, y_test)

    if model == "all" or model == "sgd":
        # SGD Classifier pipeline
        sgd_pipeline = Pipeline(
            [
                ("StandardScaler", StandardScaler()),
                ("SGDClassifier", sgd(max_iter=100, tol=1e-4, loss="log_loss")),
            ]
        )
        sgd_pipeline.fit(X, y)

        print("---SGD PREDICTION---")
        predictions, probability = sgd_pipeline.predict(
            X_test
        ), sgd_pipeline.predict_proba(X_test)
        print_predictions(predictions, probability, y_test)


def main():
    model, plot = get_args()
    df = get_iris()
    if plot:
        make_plots(df)
    make_predictions(df, model)


if __name__ == "__main__":
    main()
