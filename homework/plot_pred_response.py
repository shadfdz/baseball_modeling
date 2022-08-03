import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix


def plot_response_by_predictors(df, response, predictors, feature_type_dict):
    """
    Plot the response and predictor depending on the data type found in feature dictionary
    :param df: dataframe
    :param response: response variable
    :param predictors: predictors
    :param feature_type_dict: dictionary containing feature type
    :return: list containing relative file paths of the plots
    """
    plot_paths = []
    for pred in predictors:
        if feature_type_dict.get(response) == "continuous":
            if feature_type_dict.get(pred) == "continuous":
                plot_paths.append(cont_resp_cont_pred(df, response, pred))
            else:
                plot_paths.append(cont_resp_cat_pred(df, response, pred))
        else:
            if feature_type_dict.get(pred) == "continuous":
                plot_paths.append(cat_resp_cont_pred(df, response, pred))
            else:
                plot_paths.append(cat_resp_cat_pred(df, response, pred))
    return plot_paths


def cat_resp_cont_pred(df, response, predictor):
    """
    Violin plot of categorical response with continuous predictors
    :param df: dataframe
    :param response: response variable
    :param predictor: predictor
    :return: file path of plot
    """
    categories = df[response].unique()
    fig = go.Figure()
    for cat in categories:
        fig.add_trace(
            go.Violin(
                x=df.loc[(df[response] == cat), response],
                y=df.loc[(df[response] == cat), predictor],
                name=cat,
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig.update_layout(
        title=response + " by " + predictor,
        xaxis_title=response,
        yaxis_title=predictor,
    )
    file_path = response + "by" + predictor + ".html"
    fig.write_html(
        file="./output/" + file_path,
        include_plotlyjs="cdn",
    )
    return file_path


def cont_resp_cat_pred(df, response, predictor):
    """
    Histogram of continuous response and categorical predicator
    :param df: dataframe
    :param response: response variable
    :param predictor: predictor variable
    :return: file path
    """
    # get categories and store in a list
    categories = df[predictor].unique()

    # store response values of each categories in a list
    hist_data = []
    for cat in categories:
        hist_data.append(df[response][(df[predictor] == cat)])

    # Create distribution plot with custom bin_size
    distribution_plot = ff.create_distplot(hist_data, categories, bin_size=0.2)
    distribution_plot.update_layout(
        title=response + " by " + predictor,
        xaxis_title=response,
        yaxis_title="Distribution",
    )
    file_path = response + "by" + predictor + ".html"
    distribution_plot.write_html(
        file="./output/" + file_path,
        include_plotlyjs="cdn",
    )
    return file_path


def cat_resp_cat_pred(df, response, predictor):
    """
    Plot confusion matrix heatmap of categorical response and categorical features
    :param df: dataframe
    :param response: response variable
    :param predictor: predictor variable
    :return: file path of plot
    """
    dum_resp_df = pd.get_dummies(df[response])
    dum_pred_df = pd.get_dummies(df[predictor])

    conf_matrix = confusion_matrix(dum_resp_df.iloc[:, 0], dum_pred_df.iloc[:, 0])

    fig = go.Figure(data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max()))

    fig.update_layout(
        title=response + " by " + predictor,
        xaxis_title=response,
        yaxis_title=predictor,
    )
    file_path = response + "by" + predictor + ".html"
    fig.write_html(
        file="./output/" + file_path,
        include_plotlyjs="cdn",
    )
    return file_path


def cont_resp_cont_pred(df, response, predictor):
    """
    Scatter plot of continuous response and continuous predictor
    :param df: dataframe
    :param response: response variable
    :param predictor: predictor variable
    :return: file path to plot
    """
    x = df[predictor]
    y = df[response]

    fig = px.scatter(x=x, y=y, trendline="ols")

    fig.update_layout(
        title=response + " by " + predictor,
        xaxis_title=predictor,
        yaxis_title=response,
    )
    file_path = response + "by" + predictor + ".html"
    fig.write_html(
        file="./output/" + file_path,
        include_plotlyjs="cdn",
    )
    return file_path


def plot_diff_with_mor(df_bins, response, pred):
    """
    Plot binned mean of response by predictors as html file
    :param df_bins: data frame of binned response and predictor
    :param response: response variable
    :param pred: predictor or feature
    :return: returns the file path of the plot
    """
    bin_cat_list = df_bins["Bin"].tolist()
    cat = []
    for num in bin_cat_list:
        cat.append(str(round(num.mid, 3)))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=cat, y=df_bins["Counts"], name="Population"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=cat,
            y=df_bins["Means"] - df_bins["PopMean"],
            name="<i>µ<sub>i</sub>-µ<sub>population</sub></i>",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=[cat[0], cat[-1]],
            y=[df_bins["PopMean"].mean(), df_bins["PopMean"].mean()],
            name="<i>µ<sub>population</sub></i>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Binned Difference with Mean of Response vs Bin <br><sup>Response,Predictor: "
        + response
        + ","
        + pred
    )

    file_path = "binned" + response + "by" + pred + ".html"
    fig.write_html(
        file="./output/" + file_path,
        include_plotlyjs="cdn",
    )
    return file_path


def plot_simple_bar(x, y, x_title="", y_title="", title="", fname=""):
    x = x
    y = y

    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y,
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    file_path = fname + ".html"
    fig.write_html(file="./output/" + file_path, include_plotlyjs="cdn")
    return file_path


def plot_correlation_matrix(df_corr, title, file):
    """
    Plots correlation matrix
    :param df_corr:
    :param title:
    :param file:
    :return: file html
    """
    z = np.around(df_corr.to_numpy(), 2)
    x = df_corr.columns.to_list()
    y = df_corr.index.to_list()
    if df_corr.columns.dtype != "object":
        x = [i for i in np.around(x, 3)]
    if df_corr.index.dtype != "object":
        y = [i for i in np.around(y, 3)]

    fig = ff.create_annotated_heatmap(
        z=z,
        colorscale="thermal",
        showscale=True,
        hoverongaps=True,
    )
    fig.update_layout(
        overwrite=True,
        title=title,
        xaxis=dict(
            ticks="",
            dtick=1,
            side="top",
            gridcolor="rgb(0, 0, 0)",
            tickvals=list(range(len(x))),
            ticktext=x,
        ),
        yaxis=dict(
            ticks="",
            dtick=1,
            ticksuffix="   ",
            tickvals=list(range(len(y))),
            ticktext=y,
        ),
    )
    # this was changes
    file_name = file + ".html"
    fig.write_html(
        file="./output/" + file_name,
        include_plotlyjs="cdn",
    )
    return file_name


def plot_roc(model_list, fpr_list, tpr_list, auc_list):
    fig = go.Figure()
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    for i in range(len(model_list)):
        auc_score = model_list[i] + " (AUC : " + str(round(auc_list[i], 5)) + ")"
        fig.add_trace(
            go.Scatter(x=fpr_list[i], y=tpr_list[i], name=auc_score, mode="lines")
        )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain="domain"),
        width=700,
        height=500,
    )
    file_name = "./output/roc_plot.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    return file_name


def plot_confusion_matrix(y_true, y_pred, labels, model):
    confusion_matrix_array = confusion_matrix(y_true, y_pred, normalize="all")
    fig = px.imshow(confusion_matrix_array, text_auto=True, x=labels, y=labels)

    fig.update_layout(title="Confusion Matrix - " + model)

    file_name = "./output/cm_ " + model + ".html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )

    return file_name
