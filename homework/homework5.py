import sys
from itertools import combinations

import bin_response_by_predictor as brp
import cat_correlation as cc
import numpy as np
import pandas as pd
import plot_pred_response as ppr
import plotly.figure_factory as ff
import pymysql
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 12)


def set_response_predictors(dataframe):
    """
    Function returns list of reponse and features that are predictors
    :param dataframe: data frame
    :return: list of response and predictors
    """
    resp = "win_lose"
    predictors = dataframe.loc[:, ~dataframe.columns.isin([resp])].columns.to_list()
    pred_filtered = []
    for pred in predictors:
        if "_id" not in pred or "date" in pred:
            pred_filtered.append(pred)
    print("Response: " + resp)
    print("Predictors:", *pred_filtered)
    return resp, pred_filtered


def set_feature_type_dict(dataframe):
    """
    Function determines if features of a df is categorical or continuous
    :param dataframe: Pandas dataframe
    :return: dictionary
    """
    feature_type_dict = {}
    for col in dataframe.columns:
        if str(dataframe[col].dtype).startswith("int"):
            if 2 <= dataframe[col].unique().size < int(len(dataframe) * 0.05):
                feature_type_dict[col] = "categorical"
            else:
                feature_type_dict[col] = "continuous"
        elif str(dataframe[col].dtype).startswith("float"):
            feature_type_dict[col] = "continuous"
        else:
            feature_type_dict[col] = "categorical"
    return feature_type_dict


def set_cat_cont_predictors(feature_type_dict, response, p):
    """
    Functions set predictors to either categorical and continuous lists
    :param feature_type_dict: dict with feature type of each pred
    :param response: reponse
    :param p: print
    :return: feature type dict
    """
    cat_predictors = []
    cont_predictors = []
    for key in feature_type_dict.keys():
        if key != response:
            if feature_type_dict.get(key) == "continuous":
                cont_predictors.append(key)
            else:
                cat_predictors.append(key)
    if p:
        print(cat_predictors)
        print(cont_predictors)
    return cat_predictors, cont_predictors


def describe_df(df):
    """
    Function describes data frame
    :param df: df
    :return: None
    """
    print("Showing Top 5 Columns")
    print(df.head())
    print(df.describe())
    print("Shape of data row{}, columns {}".format(df.shape[0], df.shape[1]))
    print("Null counts for each column")
    print(df.isnull().sum())


def create_html_file(df, file_name):
    """
    Function creates html file
    :param df:
    :param file_name:
    :return: None
    """
    df_html = df.to_html(index=False, escape=False)
    file = open("../output/" + file_name + ".html", "w")
    file.write(df_html)
    file.close()


def show_var_rankings(df, resp, pred):
    """
    Function prints variable rankings
    :param df:
    :param resp:
    :param pred:
    :return: pd.Series
    """
    X = df[pred].values
    y = df[resp].values
    # Random Forest
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X, y)
    # get Variable Ranking
    feature_imp = pd.Series(clf.feature_importances_, index=pred).sort_values(
        ascending=False
    )
    return feature_imp


def get_correlation(df, pred1, pred2, feature_type_dict):
    """
    Function gets correlation based on feature type dict
    :param df:
    :param pred1:
    :param pred2:
    :param feature_type_dict:
    :return: correlation value
    """

    if (feature_type_dict.get(pred1) == "continuous") and (
        feature_type_dict.get(pred2) == "continuous"
    ):
        corr = stats.pearsonr(df[pred1], df[pred2])[0]
    elif (feature_type_dict.get(pred1) == "categorical") and (
        feature_type_dict.get(pred2) == "categorical"
    ):
        corr = cc.cat_correlation(df[pred1], df[pred2])
    else:
        corr = cc.cat_cont_correlation_ratio(df[pred1], df[pred2])
    return corr


def get_correlation_metrics(df, response, list1_pred, list2_pred, feature_type_dict):
    """
    Modified for this hw, returns correlation metrics
    :param df:
    :param response:
    :param list1_pred:
    :param list2_pred:
    :param feature_type_dict:
    :return: df of correlation metrics with file links to the plot of each predictor
    """
    corr_list = []
    for pred_outer in list1_pred:
        for pred_inner in list2_pred:
            if pred_outer != pred_inner:
                df_corr = df[(df[pred_inner].notnull()) & df[pred_outer].notnull()]
                df_corr = df_corr.reset_index()
                file_link1 = response + "by" + pred_outer + ".html"
                file_link2 = response + "by" + pred_inner + ".html"
                temp_corr_row = [
                    '<a href="{}">{}</a>'.format(file_link1, pred_outer),
                    '<a href="{}">{}</a>'.format(file_link2, pred_inner),
                    stats.pearsonr(df[pred_outer], df[pred_inner])[0],
                ]
                corr_list.append(temp_corr_row)
    col_names = ["Pred1", "Pred2", "Correlation"]
    corr_list_df = pd.DataFrame(corr_list, columns=col_names)
    corr_list_df = corr_list_df.sort_values(by="Correlation")
    return corr_list_df


def get_correlation_matrix(df, list1_pred, list2_pred, feature_type_dict):
    """
    Creates a correlation matrix between two features
    :param df:
    :param list1_pred:
    :param list2_pred:
    :param feature_type_dict:
    :return: df as a matrix
    """

    corr_matrix_df_list = []
    for pred_outer in list1_pred:
        corr_matrix_temp_list = []
        for pred_inner in list2_pred:
            df_corr = df[(df[pred_inner].notnull()) & df[pred_outer].notnull()]
            df_corr = df_corr.reset_index()
            corr_matrix_temp_list.append(
                get_correlation(df_corr, pred_outer, pred_inner, feature_type_dict)
            )
        corr_matrix_df_list.append(corr_matrix_temp_list)
    return pd.DataFrame(corr_matrix_df_list, columns=list2_pred, index=list1_pred)


def plot_correlation_matrix(df_corr, title, file):
    """
    Plots correlation matrix
    :param df_corr:
    :param title:
    :param file:
    :return: file html
    """
    z = np.around(df_corr.to_numpy(), 4)
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
    file_name = "../output/" + file + ".html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    file_link = '<a href="{}">{}</a>'.format(file_name, title)
    return file_link


def get_bin_intervals(list_1):
    """
    function returns a list of intervals from list
    :param list_1: list of stepped values
    :return: list of intervals
    """
    temp_list = []
    for val in range(len(list_1) - 1):
        j = pd.Interval(round(list_1[val], 3), round(list_1[val + 1], 3))
        temp_list.append(j)

    return temp_list


def get_df_as_matrix(
    df, list_pred1, list_pred2, category, attribute, midpoint1=False, midpoint2=False
):
    """
    Pivots a df into a matrix. Midpoint true shows interval center
    :param df: Pandas df
    :param list_pred1: columns categories
    :param list_pred2: columns categories
    :param category: measure
    :param attribute:
    :param midpoint1:
    :param midpoint2:
    :return:
    """
    list1 = list_pred1
    list2 = list_pred2
    df_list = []
    for outer_bin in list1:
        temp_list = []
        for inner_bin in list2:
            bin_array = str(outer_bin) + "," + str(inner_bin)
            val = df.loc[df[category] == bin_array, attribute]
            if val.size == 0:
                temp_list.append(0)
            else:
                temp_list.append(val.iloc[0])
        df_list.append(temp_list)

    if midpoint1:
        list1 = []
        for interval in list_pred1:
            list1.append(interval.mid)

    if midpoint2:
        list2 = []
        for interval in list_pred2:
            list2.append(interval.mid)

    return pd.DataFrame(df_list, columns=list2, index=list1)


def main():

    # Enter connection arguments in pymysql.connect()
    connection = pymysql.connect(user="guest", password="squidgames", db="baseball")
    cursor = connection.cursor()
    query = "Select * from baseball_stats;"
    df = pd.read_sql(query, connection)
    cursor.close()

    # Get predictor and response
    resp, pred = set_response_predictors(df)

    # Get cat and continuous predictors
    feature_type_dict = set_feature_type_dict(df)
    cat_predictors, cont_predictors = set_cat_cont_predictors(
        feature_type_dict, resp, True
    )

    # describe data
    describe_df(df)
    # dropping null since about < 1%
    df_processed = df.dropna(axis=0)

    # Plot each variable
    predictor_plot = ppr.PlotPredictorResponse(df_processed, feature_type_dict)
    predictor_plot.plot_response_by_predictors(resp, pred)

    # Variable Rankings
    # encode response
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(df_processed[resp])
    df_processed[resp] = label_encoder.transform(df_processed[resp])

    var_rank = show_var_rankings(df_processed, resp, pred)
    print(var_rank)

    # Get Correlation Metrics
    # plot each predictor along with response
    plot_pred = ppr.PlotPredictorResponse(df_processed, feature_type_dict)
    plot_pred.plot_response_by_predictors(resp, pred)

    # get correlation metrics for each continuous predictor combinations
    cont_correlation_df = get_correlation_metrics(
        df_processed, resp, cont_predictors, cont_predictors, feature_type_dict
    )

    # creat html file
    create_html_file(cont_correlation_df, "corr_table")

    # Getting Near Zero Variance and High correlates!!!!
    # droppings some columns
    df_processed = df_processed.drop(
        axis=1,
        labels=[
            "O_StolenBaseBB",
            "StolenBaseBB",
            "O_FIP",
            "O_SP_KBB",
            "O_SP_WHIP",
            "O_StrikeWalk",
            "series_streak",
            "O_HitOuts",
            "StrikeAtBat",
            "TeamStrikeAtBat",
            "StrikeWalk",
            "SP_WHIP",
            "SP_KBB",
            "home_streak",
        ],
    )

    # create new feature type dict without dropped columns and new features
    feature_type_dict = set_feature_type_dict(df_processed)
    resp, pred = set_response_predictors(df_processed)
    cat_predictors, cont_predictors = set_cat_cont_predictors(
        feature_type_dict, resp, False
    )
    df_processed_model = df_processed.copy()

    # create log model
    X = df_processed_model[pred].values
    y = df_processed_model[resp].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    # adjust log arg n_jobs for faster
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    y_pred = log_model.predict(X_test)
    print("Accuracy Score of Log Reg: {:.2f}".format(accuracy_score(y_test, y_pred)))

    # create Random Forest
    # https: // machinelearningmastery.com / random - forest - ensemble - in -python /
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred)
    print("Accuracy of RandomForestClassifier: {:.3f}".format(rf_acc))

    # generate correlation matrix for each cont cont predictor
    corr_df_list = []
    # iterate through predictors type list and plot correlation matrix
    corr_df_list.append(
        [
            plot_correlation_matrix(
                get_correlation_matrix(
                    df_processed, cont_predictors, cont_predictors, feature_type_dict
                ),
                "Continuous",
                "cont_cont_pred",
            )
        ]
    )

    # create a df and save into file as table with html links
    corr_matrix_pred_df = pd.DataFrame(
        corr_df_list, columns=["Correlation Plot by Predictor Type"]
    )
    create_html_file(corr_matrix_pred_df, "corr_matrix_tables")

    # Brute force Tables
    response_bins = brp.BinResponseByPredictor(df_processed, feature_type_dict)

    # brute force table for cont predictors
    brute_force_table_df = []
    title = "Link"
    for pred in combinations(cont_predictors, 2):
        df_bins, bin_list1, bin_list2 = response_bins.bin_2d_response_by_predictors(
            resp, pred[0], pred[1], 8
        )
        int_1 = get_bin_intervals(bin_list1)
        int_2 = get_bin_intervals(bin_list2)
        file_name = pred[0] + pred[1] + "response"
        file_link = plot_correlation_matrix(
            get_df_as_matrix(df_bins, int_1, int_2, "Bin", "RespBinMean", True, True),
            title,
            file_name,
        )
        brute_force_table_df.append([pred[0], pred[1], file_link])

    brute_force_html_df = pd.DataFrame(
        brute_force_table_df, columns=["Pred1", "Pred2", "Combined Response Plot"]
    )
    create_html_file(brute_force_html_df, "brute_force_tables")

    # Batting average seems to be a good indicator of win/lose
    # But there is an error I think in my split since this is time series
    # I would split my test to 20% rows of most recent data!
    # I didnt have time to tune either models but they both have the same results
    # I'm assuming I have a nostradamus variable in here with the way I set up the data


if __name__ == "__main__":
    sys.exit(main())
