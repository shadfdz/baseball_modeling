# HW 1
# Shad Fernandez
# BDA 696 ML Engineering
# 11-SEP-2021

# import necessary modules
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# read iris dataframe
iris_df = pd.read_csv(
    "../datasets/iris.data",
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
)


# create numpy array without categorical variables
iris_array = np.array(iris_df.iloc[1:, 0:4])

# explore data set, update quantiles, use column names for species?
print("mean: ", np.round(iris_array.mean(axis=0), 2))
print("stdev: ", np.round(iris_array.std(axis=0), 2))
print("max: ", iris_array.max(axis=0))
print("min", iris_array.min(axis=0))
print(np.quantile(iris_array, 0.5, axis=0))


# Box plot of sepal length
iris_plot = px.box(iris_df, x="class", y="sepal_length")
iris_plot.update_layout(title_text="Sepal Length Distribution by Class")
iris_plot.show()


# scatter of sepal length and width
iris_scatter = px.scatter(
    iris_df,
    x="sepal_length",
    y="sepal_width",
    color="class",
    title="Sepal Lengths and Widths of Iris Classes",
)
iris_scatter.show()

# Bar chart of each measure mean
class_features = iris_df.columns.to_list()[0:4]
iris_fig_bar = go.Figure(
    data=[
        go.Bar(
            name="Setosa",
            x=class_features,
            y=iris_df.loc[
                iris_df["class"] == "Iris-setosa", "sepal_length":"petal_width"
            ].mean(),
        ),
        go.Bar(
            name="Versicolor",
            x=class_features,
            y=iris_df.loc[
                iris_df["class"] == "Iris-versicolor", "sepal_length":"petal_width"
            ].mean(),
        ),
        go.Bar(
            name="Virginica",
            x=class_features,
            y=iris_df.loc[
                iris_df["class"] == "Iris-virginica", "sepal_length":"petal_width"
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


# violin plot
# iris_fig_vi = go.Figure()
#
# # for features in class_features:
# #     iris_fig_vi.add_trace(go.Violin(x=iris_df['class'],y=iris_df[features]))
#
# iris_fig_vi.show()
