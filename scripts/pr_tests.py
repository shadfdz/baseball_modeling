import math
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api
from plotly.subplots import make_subplots
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# make a directory to hold plot html files if it doesnt exist
if not os.path.exists("../assignment4Plots"):
    os.makedirs("../assignment4Plots")


# function to check if a pandas column is categorical or continuous
def is_continuous(df, column_name):
    # check to see if column is int or float and has more than 2 unique values
    if (
        df[column_name].apply(isinstance, args=[(int, float)]).all()
        and df[column_name].nunique() > 2
    ):
        return True

    # else, column is not continuous
    else:
        return False


# function to plot continuous predictors
def plot_continuous(df, series, response):
    # if response is continuous, scatterplot with trend line
    if is_continuous(df, response):
        fig = px.scatter(df, x=series.name, y=response, trendline="ols")

    # if response is categorical, violin plot
    else:
        fig = px.violin(df, x=response, y=series.name, box=True)
    return fig


# function to plot categorical predictors
def plot_categorical(df, series, response):
    # if response is continuous, violin plot grouped by predictor
    if is_continuous(df, response):
        fig = px.violin(df, x=series.name, y=response, box=True)

    # if response is categorical, heatplot
    else:
        fig = px.density_heatmap(df, x=series.name, y=response)
    return fig


# this function performs either a linear or logistic regression (based on response being categorical or continuous)
# Parameters:
# df = pandas dataframe
# col = string name of column to perform regression on
# response = string name of response variable
# RETURNS:
# fig = plot of linear or logistic regression
# t-value = t value of regression
# p-value = p value of regression
def regression(df, col, response):
    # if response is continuous, linear regression
    if is_continuous(df, response):
        predictor = statsmodels.api.add_constant(df[col].values)
        linear_regression_model = statsmodels.api.OLS(df[response].values, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {col}")
        print(linear_regression_model_fitted.summary())

        # get t and p value
        t_value = round(linear_regression_model_fitted.tvalues[1], 4)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

        # plot the linear regression
        fig = px.scatter(x=df[col].values, y=df[response].values, trendline="ols")
        fig.update_layout(
            title=f"Variable: {col}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {col}",
            yaxis_title=response,
        )

    # if response is categorical (binary), logistic regression
    else:
        predictor = statsmodels.api.add_constant(df[col].values)
        logistic_regression_model = statsmodels.api.Logit(
            df[response].values, predictor
        )
        logistic_regression_model_fitted = logistic_regression_model.fit()
        print(f"Variable: {col}")
        print(logistic_regression_model_fitted.summary())

        # get t and p value
        t_value = round(logistic_regression_model_fitted.tvalues[1], 4)
        p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

        yhat = logistic_regression_model_fitted.predict(predictor)

        # plot the logistic regression
        fig1 = px.scatter(x=df[col].values, y=df[response].values)
        fig1.update_layout(
            title=f"Variable: {col}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {col}",
            yaxis_title=response,
        )
        fig2 = px.scatter(x=df[col], y=yhat, color_discrete_sequence=["red"])
        fig = go.Figure(data=fig1.data + fig2.data)
        fig.update_layout(
            title=f"Variable: {col}: (t-value={t_value}) (p-value={p_value}) with predicted probabilities in red",
            xaxis_title=f"Variable: {col}",
            yaxis_title=response,
        )
    return fig, t_value, p_value


# this function will invoke the difference with mean of response formula as well as the plot
# parameters:
# df = dataframe
# predictor = string (name of predictor)
# response = string (name of response)
# weighted = Boolean (True if using weighted difference, false otherwise)
# RETURNS:
# result = numeric value of difference with mean of response
# fig = plot of difference with mean of response
def diff_mean_response(df, predictor, response, weighted):

    # first, need to create the bins, but this will be different based on if predictor is continuous or not:
    # if continuous, then # of bins is the square root of the total number of observations
    # if not continuous, then categorical, so bins will be # of categories - 1 and bin each category in own bin
    if is_continuous(df, predictor):
        holder, bin_create = np.histogram(
            df[predictor], bins=math.ceil(np.sqrt(len(df[predictor])))
        )
    else:
        holder, bin_create = np.histogram(
            df[predictor], bins=len(np.unique(df[predictor])) - 1
        )

    # next, get the bin number of each observation (array of bin numbers corresponding to each predictor observation)
    bin_numbers = np.digitize(df[predictor], bin_create)

    # create an array containing tuples with the following structure:
    # (x,y), where:
    # x = response value corresponding to predictor value at i
    # y = corresponding bin number for the response value
    res = [(df[response].values[i], bin_numbers[i]) for i in range(len(bin_numbers))]

    # sort array by bin number
    sorted_bins = sorted(res, key=lambda x: x[1])

    # calculate the average response value of each bin, and store in bin_response_means
    # where index 0 = bin 1, index 1 = bin 2, etc.
    bin_response_means = []
    for i in np.unique(bin_numbers):
        arr = []
        for j in sorted_bins:
            if j[1] == i:
                arr.append(j[0])
            else:
                continue
        bin_response_means.append(np.mean(arr))

    # get population mean of response
    pop_mean = np.mean(df[response])

    # count the number of values in each bin
    bin_count = np.bincount(bin_numbers)
    bin_count = np.delete(bin_count, 0)

    # get the total # of observations in the population (equivalent to len(df[predictor]))
    total_population = np.sum(bin_count)

    # now, calculate the difference with mean of response using the formula
    # if weighted = True, do weighted calculation of diff of response means,
    # if weighted = False, do unweighted calculation
    running_sum = 0
    diff_bin_pop = []
    # use this index to keep track of the current bin
    bin_idx = 0
    for i in bin_response_means:
        if weighted:
            pop_proportion = (
                bin_count[bin_idx] / total_population
            )  # population proportion for bin i
            running_sum += pop_proportion * ((i - pop_mean) ** 2)
            bin_idx += 1

        else:
            running_sum += (i - pop_mean) ** 2
        diff_bin_pop.append(i - pop_mean)

    # divide running sum by the number of bins, which is the max of all the bin numbers to get final result
    result = running_sum / max(np.unique(bin_numbers))

    # now plot this
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Histogram(
            histfunc="sum",
            x=bin_create,
            y=bin_count,
            xbins=dict(
                start=bin_create[0],
                end=bin_create[-1],
                size=bin_create[1] - bin_create[0],
            ),
            autobinx=False,
            name="Population",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=bin_create,
            y=diff_bin_pop,
            name=r"$\mu_{i} - \mu_{pop}$",
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.add_hline(y=pop_mean, secondary_y=True, name=r"$\mu_{pop}$")

    fig.update_xaxes(title_text="Predictor Bin")
    fig.update_yaxes(title_text="<b>Population</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Response</b>", secondary_y=True)
    fig.update_layout(title=f"Variable: {predictor}")

    return result, fig


# function to get the random forest variable importance
# parameters:
# X_train: array of all feature values
# y_train: array of all response values
# ** Returns **
# list of numbers pertaining to the importance of each predictor
def random_forest_variable_importance(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf.feature_importances_


# main Parameters:
# dataframe: pandas dataframe
# predictors: list of strings containing the names of the predictors in "dataframe"
# reponse: name of the response variable in a string
# ** Returns **
# "finalTable.html" in assignments folder which contains all graphs and values
def main(dataframe, predictors, response):
    # create a random seed for reproducible results (especially for random forest)
    np.random.seed(100)

    # beacause the random forest variable importance computes the ranking all at once,
    # do this first, then pull the value when looping through the predictors
    rand_forest_ranking = random_forest_variable_importance(
        dataframe.loc[:, predictors].values, dataframe[response].values
    )
    # loop through predictors, and display correct plot
    idx = 0
    eda_plots = []
    reg_plots = []
    diff_plots = []
    t_values = []
    p_values = []
    forest_rankings = []
    diff_weighted = []
    diff_unweighted = []
    folder = "../assignment4Plots/"
    # file_prefix = "file://"
    for i in predictors:
        # store strings for all the plots
        eda_plot_string = "eda_" + i.replace(" ", "") + ".html"
        reg_plot_string = "reg_" + i.replace(" ", "") + ".html"
        diff_plot_string = "diff_" + i.replace(" ", "") + ".html"
        # CONDITION WHERE PREDICTOR IS CONTINUOUS
        # 1) plot predictor vs response using correct plot
        # 2) show p-value and t-score (continuous response: linear regression, categorical response: logistic regression
        # 3) difference with mean of response along with plot (weighted and unweighted)
        # 4) Random Forest var importance ranking
        if is_continuous(dataframe, i):
            eda_fig = plot_continuous(dataframe, dataframe[i], response)
            reg_fig, t_val, p_val = regression(dataframe, i, response)
            # append t values and p values to list
            t_values.append(t_val)
            p_values.append(p_val)
            i_diff_unweighted, plot = diff_mean_response(dataframe, i, response, False)
            i_diff_weighted, _ = diff_mean_response(dataframe, i, response, True)
            # append these values to correct diff list
            diff_unweighted.append(i_diff_unweighted)
            diff_weighted.append(i_diff_weighted)

            # get random forest ranking and append to list
            i_forest_ranking = rand_forest_ranking[idx]
            forest_rankings.append(i_forest_ranking)

            # write html files of figures, then append to a list of each
            project_path = "/BDA_696/"
            eda_fig.write_html(folder + eda_plot_string)
            reg_fig.write_html(folder + reg_plot_string)
            plot.write_html(folder + diff_plot_string)
            eda_plots.append(project_path + "/assignment4Plots/" + eda_plot_string)
            reg_plots.append(project_path + "/assignment4Plots/" + reg_plot_string)
            diff_plots.append(project_path + "/assignment4Plots/" + diff_plot_string)

        # CONDITION WHERE PREDICTOR IS CATEGORICAL
        # 1) plot predictor vs response using correct plot
        # 2) Difference with mean of response along with plot (weighted and unweighted)
        else:
            eda_fig = plot_categorical(dataframe, dataframe[i], response)
            i_diff_unweighted, plot = diff_mean_response(dataframe, i, response, False)
            i_diff_weighted, _ = diff_mean_response(dataframe, i, response, True)
            # append to correct list
            diff_unweighted.append(i_diff_unweighted)
            diff_weighted.append(i_diff_weighted)

            # write html files of plots and append to correct list
            eda_fig.write_html(folder + eda_plot_string)
            plot.write_html(folder + diff_plot_string)
            eda_plots.append(project_path + "/assignment4Plots/" + eda_plot_string)
            diff_plots.append(project_path + "/assignment4Plots/" + diff_plot_string)

            # append null for t values, p values, and random Forest rankings as these are categorical predictors
            t_values.append(None)
            p_values.append(None)
            forest_rankings.append(None)
            forest_rankings.append(None)

        idx += 1

    # now, create dataframe with all of the gathered information
    result = pd.DataFrame(
        list(
            zip(
                predictors,
                eda_plots,
                t_values,
                p_values,
                reg_plots,
                diff_weighted,
                diff_unweighted,
                diff_plots,
                forest_rankings,
            )
        ),
        columns=[
            "Predictor",
            "EDA Plots",
            "t-value",
            "p-value",
            "Regression Plots",
            "Weighted DMR",
            "Unweighted DMR",
            "DMR Plot",
            "RF Variable Importance",
        ],
    )
    result.to_html(
        "finalTable.html",
        formatters={
            "EDA Plots": lambda x: f'<a href="{x}">{x}</a>',
            "Regression Plots": lambda x: f'<a href="{x}">{x}</a>',
            "DMR Plot": lambda x: f'<a href="{x}">{x}</a>',
        },
        escape=False,
    )
    return


if __name__ == "__main__":
    # main function takes in pandas dataframe, list of predictors, and response
    # produces "finalTable.html" file which can be opened on a browser
    # this file is located in the "Assignments" folder

    # below is an example ran on the breast cancer dataset provided by sklearn
    data = datasets.load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.options.display.max_colwidth = 100
    main(df, data.feature_names, "target")
