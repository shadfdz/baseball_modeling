import sys
from itertools import combinations, combinations_with_replacement

import bin_response_by_predictor as brp
import cat_correlation as cc
import numpy as np
import pandas as pd
import plot_pred_response as ppr
import plotly.figure_factory as ff
from scipy import stats

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 12)


def set_response_predictors(dataframe):
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


def set_cat_cont_predictors(feature_type_dict, response):
    cat_predictors = []
    cont_predictors = []
    for key in feature_type_dict.keys():
        if key != response:
            if feature_type_dict.get(key) == "continuous":
                cont_predictors.append(key)
            else:
                cat_predictors.append(key)
    return cat_predictors, cont_predictors


def get_correlation(df, pred1, pred2, feature_type_dict):

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
                    get_correlation(df_corr, pred_outer, pred_inner, feature_type_dict),
                ]
                corr_list.append(temp_corr_row)
    col_names = ["Pred1", "Pred2", "Correlation"]
    corr_list_df = pd.DataFrame(corr_list, columns=col_names)
    return corr_list_df


def get_correlation_matrix(df, list1_pred, list2_pred, feature_type_dict):

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
    z = np.around(df_corr.to_numpy(), 4)
    x = df_corr.columns.to_list()
    y = df_corr.index.to_list()
    if df_corr.columns.dtype != "object":
        x = ["'" + (str(i) + "'") for i in np.around(x, 3)]
    if df_corr.index.dtype != "object":
        y = ["'" + (str(i) + "'") for i in np.around(y, 3)]

    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        annotation_text=z,
        colorscale="thermal",
        showscale=True,
        hoverongaps=True,
    )
    fig.update_layout(
        title=title,
    )
    file_name = "../output/" + file + ".html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    file_link = '<a href="{}">{}</a>'.format(file_name, title)
    fig.show()
    return file_link


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


def main():
    # Retrieve data
    df = pd.read_csv("../datasets/Titanic.csv")
    df = df.drop(["PassengerId", "Ticket"], axis=1)
    df["Name"] = df["Name"].str.split(",").str.get(0)

    # Get list of response and predictors
    response, predictors = set_response_predictors(df)

    # Split predictors to categorical and continuous and add to respective list
    feature_type_dict = set_feature_type_dict(df)
    cat_predictors, cont_predictors = set_cat_cont_predictors(
        feature_type_dict, response
    )

    # PART 1
    # plot each predictor along with response
    plot_pred = ppr.PlotPredictorResponse(df, feature_type_dict)
    plot_pred.plot_response_by_predictors(response, predictors)

    # get correlation metrics for each continuous, categorical, and cont/cat predictor combinations
    cont_correlation_df = get_correlation_metrics(
        df, response, cont_predictors, cont_predictors, feature_type_dict
    )
    cat_correlation_df = get_correlation_metrics(
        df, response, cat_predictors, cont_predictors, feature_type_dict
    )
    cat_cont_pred_correlation_df = get_correlation_metrics(
        df, response, cat_predictors, cat_predictors, feature_type_dict
    )

    # merge to one table and export as html
    corr_df = pd.concat(
        [cont_correlation_df, cat_correlation_df, cat_cont_pred_correlation_df]
    ).sort_values(by="Correlation", ascending=False)

    # creat html file
    corr_df = corr_df.to_html(index=False, escape=False)
    file = open("../output/corr_table.html", "w")
    file.write(corr_df)
    file.close()

    # # PART 2
    # # generate correlation matrix for each cont cont predictor
    predictor_type_list = [cat_predictors, cont_predictors]
    corr_plot_title_list = [
        "Categorical",
        "Categorical/Continuous",
        "Continuous",
    ]
    corr_plot_file_name_list = [
        "cat_cat_pred",
        "cat_cont_pred",
        "cont_cont_pred",
    ]
    i = 0
    corr_df_list = []
    for comb in combinations_with_replacement(predictor_type_list, 2):
        corr_df_list.append(
            [
                plot_correlation_matrix(
                    get_correlation_matrix(df, comb[0], comb[1], feature_type_dict),
                    corr_plot_title_list[i],
                    corr_plot_file_name_list[i],
                )
            ]
        )
        i += 1

    corr_matrix_pred_df = pd.DataFrame(
        corr_df_list, columns=["Correlation Plot by Predictor Type"]
    )
    corr_matrix_pred_df = corr_matrix_pred_df.to_html(index=False, escape=False)
    file = open("../output/corr_matrix_tables.html", "w")
    file.write(corr_matrix_pred_df)
    file.close()

    # Part 3
    response_bins = brp.BinResponseByPredictor(df, feature_type_dict)
    #
    # brute force table for cont predictors
    for comb in combinations(cont_predictors, 2):
        df_bins, bin_list1, bin_list2 = response_bins.bin_2d_response_by_predictors(
            response, comb[0], comb[1], 8
        )
        int_1 = get_bin_intervals(bin_list1)
        int_2 = get_bin_intervals(bin_list2)
        title = comb[0] + " and " + comb[1] + " Correlation Matrix"
        file_name = comb[0] + comb[1] + "corrplot"
        plot_correlation_matrix(
            get_df_as_matrix(df_bins, int_1, int_2, "Bin", "RespBinMean", True, True),
            title,
            file_name,
        )

    for pred in combinations(cat_predictors, 2):
        print(pred[0], pred[1])
        df_bins, bin_list1, bin_list2 = response_bins.bin_2d_cat_cat_pred(
            response, pred[0], pred[1], 8
        )
        test = get_df_as_matrix(
            df_bins, bin_list1, bin_list2, "Bin", "RespBinMean", False, False
        )
        title = pred[0] + pred[1]
        file = pred[0] + pred[1]
        plot_correlation_matrix(test, title, file)

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
