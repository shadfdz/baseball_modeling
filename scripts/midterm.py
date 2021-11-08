import sys
from itertools import combinations

import bin_response_by_predictor as brp
import cat_correlation as cc
import numpy as np
import pandas as pd
import plot_pred_response as ppr
import plotly.figure_factory as ff
from scipy import stats

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 12)


def get_response_predictors(dataframe):
    """
    Prompts user to enter response name. Function returns list of reponse and features that are predictors
    :param dataframe: data frame
    :return: list of response and predictors
    """
    # print("The following are features of the dataset: ")
    # print(*dataframe.columns)
    # response_feature = str(
    #     input("\nPlease enter the column name of the response variable: ")
    # )
    resp = "Survived"
    preds = dataframe.loc[:, ~dataframe.columns.isin([resp])].columns.to_list()
    print("Response: " + resp)
    print("Predictors:", *preds)
    return resp, preds


def get_feature_type_dict(dataframe):
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


def get_cat_cont_predictor_list(feature_type_dict, response):
    cat_predictors = []
    cont_predictors = []
    for key in feature_type_dict.keys():
        if key != response:
            if feature_type_dict.get(key) == "continuous":
                cont_predictors.append(key)
            else:
                cat_predictors.append(key)
    return cat_predictors, cont_predictors


def get_correlation(df, pred1, pred2, corr_type):
    corr = 0
    if corr_type == "pearson":
        corr = stats.pearsonr(df[pred1], df[pred2])[0]
    elif corr_type == "categorical":
        corr = cc.cat_correlation(df[pred1], df[pred2])
    elif corr_type == "ratio":
        corr = cc.cat_cont_correlation_ratio(df[pred1], df[pred2])
    else:
        print("Correlation category does not exist")
    return corr


def get_corr_metrics(
    df, response, feature_type_dict, list1_pred, list2_pred, corr_function
):
    resp_pred_plotter = ppr.PlotPredictorResponse(df, feature_type_dict)
    corr_list = []
    for pred_outer in list1_pred:
        for pred_inner in list2_pred:
            if pred_outer != pred_inner:
                df_corr = df[(df[pred_inner].notnull()) & df[pred_outer].notnull()]
                df_corr = df_corr.reset_index()
                temp_corr_row = [
                    '=HYPERLINK("'
                    + resp_pred_plotter.plot_response_by_predictors(
                        response, [pred_outer]
                    )
                    + '","'
                    + pred_outer
                    + '")',
                    '=HYPERLINK("'
                    + resp_pred_plotter.plot_response_by_predictors(
                        response, [pred_inner]
                    )
                    + '","'
                    + pred_inner
                    + '")',
                    get_correlation(df_corr, pred_outer, pred_inner, corr_function),
                ]
                corr_list.append(temp_corr_row)
    return corr_list


def get_correlation_matrix(df, list1_pred, list2_pred, corr_function):

    corr_matrix_df_list = []
    for pred_outer in list1_pred:
        corr_matrix_temp_list = []
        for pred_inner in list2_pred:
            df_corr = df[(df[pred_inner].notnull()) & df[pred_outer].notnull()]
            df_corr = df_corr.reset_index()
            corr_matrix_temp_list.append(
                get_correlation(df_corr, pred_outer, pred_inner, corr_function)
            )
        corr_matrix_df_list.append(corr_matrix_temp_list)
    df_corr = pd.DataFrame(corr_matrix_df_list, columns=list2_pred, index=list1_pred)
    z = np.around(df_corr.to_numpy(), 4)
    x = df_corr.columns.to_list()
    y = df_corr.index.to_list()
    return df_corr, x, y, z


def plot_correlation_matrix(x, y, z, title, file):

    fig = ff.create_annotated_heatmap(
        z=z, x=x, y=y, annotation_text=z, colorscale="thermal", showscale=True
    )
    fig.update_layout(
        title=title,
    )
    file_name = file
    fig.write_html(
        file="../output/" + file_name,
        include_plotlyjs="cdn",
    )
    fig.show()


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


def get_brute_force_table(
    df, feature_type_dict, response, pred1, pred2, midpoint1=False, midpoint2=False
):
    df_matrix = 0
    response_bins = brp.BinResponseByPredictor(df, feature_type_dict)
    for pred_outer in pred1:
        for pred_inner in pred2:
            if pred_outer != pred_inner:
                (
                    df_bins,
                    bin_list1,
                    bin_list2,
                ) = response_bins.bin_2d_response_by_predictors(
                    response, pred_outer, pred_inner, 8
                )
                df_matrix = get_df_as_matrix(
                    df_bins,
                    bin_list1,
                    bin_list2,
                    "Bin",
                    "RespBinMean",
                    midpoint1,
                    midpoint2,
                )
    return df_matrix


def create_sheet(temp_list, col_names, excel_instance, file_name):
    """
    function adds sheet to instance of xlsxwriter and returns a df
    :param temp_list: list of lists to create into pd df
    :param col_names: list of col names
    :param excel_instance: instance of xlsxwriter
    :param file_name: file name
    :return:
    """
    correlation_df = pd.DataFrame(temp_list, columns=col_names)
    correlation_df = correlation_df.sort_values(by="Correlation", ascending=False)
    correlation_df.to_excel(excel_instance, sheet_name=file_name, index=False)
    return correlation_df


def main():
    # get data
    df = pd.read_csv("../datasets/Titanic.csv")
    df = df.drop(["PassengerId", "Ticket"], axis=1)

    df["Name"] = df["Name"].str.split(",").str.get(0)

    # get list of response and predictors
    response, predictors = get_response_predictors(df)

    # split predictors to categorical and continuous and add to respective list
    feature_type_dict = get_feature_type_dict(df)
    cat_predictors, cont_predictors = get_cat_cont_predictor_list(
        feature_type_dict, response
    )

    # PART 1
    # print correlation metrics for each continuous predictor
    list1 = get_corr_metrics(
        df, response, feature_type_dict, cont_predictors, cont_predictors, "pearson"
    )
    list2 = get_corr_metrics(
        df, response, feature_type_dict, cat_predictors, cont_predictors, "ratio"
    )
    list3 = get_corr_metrics(
        df, response, feature_type_dict, cat_predictors, cat_predictors, "categorical"
    )
    #
    print(list1)
    print(list2)
    print(list3)

    # PART 2
    # generate correlation matrix for each cont cont predictor
    df_corr_cont, x, y, z = get_correlation_matrix(
        df, cont_predictors, cont_predictors, "pearson"
    )
    plot_correlation_matrix(
        x, y, z, "Correlation Matrix Continuous Predictors", "cont_cont"
    )

    df_corr_cat_cont, x, y, z = get_correlation_matrix(
        df, cat_predictors, cont_predictors, "ratio"
    )
    plot_correlation_matrix(
        x, y, z, "Correlation Matrix Categorical and Continuous Predictors", "cat_cont"
    )

    df_corr_cat, x, y, z = get_correlation_matrix(
        df, cat_predictors, cat_predictors, "categorical"
    )
    plot_correlation_matrix(
        x, y, z, "Correlation Matrix Categorical Predictors", "cat_cat"
    )

    # # Part 3
    response_bins = brp.BinResponseByPredictor(df, feature_type_dict)

    # brute force table for cont predictors
    for comb in combinations(cont_predictors, 2):
        print(comb)
        df_bins, bin_list1, bin_list2 = response_bins.bin_2d_response_by_predictors(
            response, comb[0], comb[1], 8
        )
        int_1 = get_bin_intervals(bin_list1)
        int_2 = get_bin_intervals(bin_list2)
        print(df_bins)
        print(get_df_as_matrix(df_bins, int_1, int_2, "Bin", "RespBinMean", True, True))

    for pred in combinations(cat_predictors, 2):
        print(pred)
        df_bins, bin_list1, bin_list2 = response_bins.bin_2d_cat_cat_pred(
            response, pred[0], pred[1], 8
        )
        print(df_bins)
        print(
            get_df_as_matrix(
                df_bins, bin_list1, bin_list2, "Bin", "RespBinMean", False, False
            )
        )

    for pred_outer in cat_predictors:
        for pred_inner in cont_predictors:
            print(pred_outer, pred_inner)
            df_bins, bin_list1, bin_list2 = response_bins.bin_2d_cat_cont_pred(
                response, pred_outer, pred_inner, 8
            )
            print(df_bins)
            int_2 = get_bin_intervals(bin_list2)
            print(
                get_df_as_matrix(
                    df_bins, bin_list1, int_2, "Bin", "RespBinMean", False, True
                )
            )


if __name__ == "__main__":

    sys.exit(main())
