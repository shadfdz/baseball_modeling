# HW 1
# Shad Fernandez
# BDA 696 ML Engineering
# 11-SEP-2021

# import necessary modules
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():

    # Iris data set source
    iris_source = "https://archive.ics.uci.edu/ml/datasets/iris"

    # read iris data and create column names
    iris_df = pd.read_csv(
        "../datasets/iris.data",
        names=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "Iris_class",
        ],
    )

    # remove 'Iris-' from class names
    iris_df["Iris_class"] = iris_df["Iris_class"].map(lambda x: x.lstrip("Iris"))
    iris_df["Iris_class"] = iris_df["Iris_class"].map(lambda x: x.lstrip("-"))
    iris_df["Iris_class"].head()

    # show data set description
    print("Iris Data Set\nSource: {}".format(iris_source))
    print("Iris Data Size: {}".format(iris_df.shape))  # print data shape
    print(
        "Number of Null Columns: {}\n".format(iris_df.columns.isnull().sum())
    )  # print columns will null
    print(
        "\t--Iris Date Set (Top 5 Rows)--\n{}".format(iris_df.head(5))
    )  # print top 5 rows
    print("\n\t--Data Description--")  # print data description
    print(iris_df.describe().round(2))

    # scatter of sepal length and width
    iris_scatter = px.scatter(
        iris_df,
        x="sepal_length",
        y="sepal_width",
        color="Iris_class",
        title="Sepal Lengths and Widths of Iris Classes",
    )
    iris_scatter.show()

    # Scatter Matrix of Iris data set
    iris_fig_scatter_matrix = px.scatter_matrix(
        iris_df,
        dimensions=iris_df.columns.tolist()[0:4],
        color="Iris_class",
        title="Scatter Matrix of Iris Data",
        opacity=0.6,
        labels=iris_df.columns.tolist()[0:4],
    )
    iris_fig_scatter_matrix.show()

    # Bar chart of each features of every Iris class
    class_features = iris_df.columns.to_list()[0:4]
    iris_fig_bar = go.Figure(
        data=[
            go.Bar(
                name="Setosa",
                x=class_features,
                y=iris_df.loc[
                    iris_df["Iris_class"] == "setosa", "sepal_length":"petal_width"
                ].mean(),
            ),
            go.Bar(
                name="Versicolor",
                x=class_features,
                y=iris_df.loc[
                    iris_df["Iris_class"] == "versicolor", "sepal_length":"petal_width"
                ].mean(),
            ),
            go.Bar(
                name="Virginica",
                x=class_features,
                y=iris_df.loc[
                    iris_df["Iris_class"] == "virginica", "sepal_length":"petal_width"
                ].mean(),
            ),
        ]
    )

    iris_fig_bar.update_layout(
        title="Mean Measurements For Each Iris Class",
        yaxis_title="(cm)",
        legend_title="Iris Class",
    )
    iris_fig_bar.show()

    # Violin chart of feature distribution of each Iris class
    iris_df_pivot = iris_df.melt(id_vars="Iris_class")
    plot_colors = ["red", "orange", "brown"]
    iris_class_types = iris_df["Iris_class"].unique().tolist()
    iris_fig_viol = go.Figure()

    for species, color in zip(iris_class_types, plot_colors):
        iris_fig_viol.add_trace(
            go.Violin(
                x=iris_df_pivot["variable"][iris_df_pivot["Iris_class"] == species],
                y=iris_df_pivot["value"][iris_df_pivot["Iris_class"] == species],
                legendgroup=species,
                scalegroup=species,
                name=species,
                line_color=color,
            )
        )
    iris_fig_viol.update_traces(box_visible=True, meanline_visible=True)
    iris_fig_viol.update_layout(
        violinmode="group",
        title="Violin Plot of Each Iris Class Feature",
        yaxis_title="cm",
    )
    iris_fig_viol.show()

    # Box plot
    iris_fig_box = px.box(iris_df_pivot, x="variable", y="value", color="Iris_class")
    iris_fig_box.update_layout(
        title="Box Plot of the Iris Data Set", yaxis_title="cm", xaxis_title=""
    )
    iris_fig_box.show()

    # Modeling Data
    # set dependent and ind variables
    x = np.array(iris_df.select_dtypes(include="float64"))
    y = iris_df["Iris_class"]

    # set seed and create test data set
    np.random.seed(1)
    test_index = np.random.choice(x.shape[0], size=4, replace=False)
    x_test = x[test_index]

    # create pipeline for select Data Models
    pipe_randforest = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier()),
        ]
    )
    pipe_log = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("LogisticRegression", LogisticRegression()),
        ]
    )

    pipe_lda = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()),
        ]
    )

    # Fit data on models
    pipe_randforest.fit(x, y)
    randforest_probability = pipe_randforest.predict_proba(x_test)
    randforest_prediction = pipe_randforest.predict(x_test)

    pipe_log.fit(x, y)
    log_probability = pipe_log.predict_proba(x_test)
    log_prediction = pipe_log.predict(x_test)

    pipe_lda.fit(x, y)
    lda_probability = pipe_lda.predict_proba(x_test)
    lda_prediction = pipe_lda.predict(x_test)

    # Print predicted probability by each model
    print("\n\t--Probability by Each Model")
    print(f"Random forest:\n{randforest_probability}")
    print(f"Log Regression:\n{log_probability}")
    print(f"LDA:\n{lda_probability}\n")

    # Print Prediction by each
    print("\t--Iris Class Prediction by Each Model")
    print(f"Test Classes: {y[test_index].tolist()}")
    print(f"Random forest: {randforest_prediction}")
    print(f"Log Regression: {log_prediction}")
    print(f"LDA: {lda_prediction}")


if __name__ == "__main__":
    main()
