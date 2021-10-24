import sys

import cat_correlation as cc
import matplotlib.pyplot as plt
import pandas as pd
import plot_pred_response as ppr
import seaborn as sns
from scipy import stats

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 12)


def get_response_predictors(dataframe):
    print("The following are features of the dataset: ")
    print(*dataframe.columns)
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
    get_col_type_dict ...
    add more to detect categories
    :param dataframe:
    :return:
    """
    feature_type_dict = {}
    for col in dataframe.columns:
        if str(dataframe[col].dtype).startswith("int"):
            if dataframe[col].unique().size >= 2 and dataframe[col].unique().size < int(
                len(dataframe) * 0.05
            ):
                feature_type_dict[col] = "categorical"
            else:
                feature_type_dict[col] = "continuous"
        elif str(dataframe[col].dtype).startswith("float"):
            feature_type_dict[col] = "continuous"
        else:
            feature_type_dict[col] = "categorical"
    return feature_type_dict


def main():
    # get data
    df = pd.read_csv("../datasets/titanic.csv")
    df = df.drop("Name", axis=1)

    # get list of response and predictors
    response, predictors = get_response_predictors(df)

    # split predictors to categorical and continuous and add to respective list
    feature_type_dict = get_feature_type_dict(df)
    cat_predictors = []
    cont_predictors = []
    for key in feature_type_dict.keys():
        if feature_type_dict.get(key) == "continuous":
            cont_predictors.append(key)
        else:
            cat_predictors.append(key)
    print(cat_predictors)
    print(cont_predictors)

    corr_temp_list = []
    corr_col_names = ["Predictor1", "Predictor2", "Correlation"]
    # # print correlation metrics for each continuous predictor
    for pred_outer in cont_predictors:
        for pred_inner in cont_predictors:
            if pred_outer != pred_inner:
                df_corr = df[(df[pred_inner].notnull()) & df[pred_outer].notnull()]
                temp_corr_row = [
                    pred_outer,
                    pred_inner,
                    stats.pearsonr(df_corr[pred_outer], df_corr[pred_inner])[0],
                ]
                corr_temp_list.append(temp_corr_row)

    # print correlation metrics for each categorical predictor
    for pred_outer in cat_predictors:
        for pred_inner in cat_predictors:
            if pred_outer != pred_inner:
                temp_corr_row = [
                    pred_outer,
                    pred_inner,
                    cc.cat_correlation(df[pred_outer], df[pred_inner]),
                ]
                corr_temp_list.append(temp_corr_row)

    # print correlation metrics for each categorical cont predictor
    for pred_outer in cat_predictors:
        for pred_inner in cont_predictors:
            if pred_outer != pred_inner:
                temp_corr_row = [
                    pred_outer,
                    pred_inner,
                    cc.cat_cont_correlation_ratio(df[pred_outer], df[pred_inner]),
                ]
                corr_temp_list.append(temp_corr_row)

    # print in table by metrics in descending order
    correlation_df = pd.DataFrame(corr_temp_list, columns=corr_col_names)
    correlation_df = correlation_df.sort_values(by="Correlation", ascending=False)
    print(correlation_df)

    # # put links to original variable plot done in hw4
    resp_pred_plotter = ppr.PlotPredictorResponse(df, feature_type_dict)
    resp_pred_plotter.plot_response_by_predictors(response, predictors)

    # WIP IN PROGRESS 24-OCT-2021
    # generate correlation matrices for the above three
    # print heat map of all three combinations

    # generate correlation matrix for each cont cont predictor
    df_corr_cont = df[cont_predictors].corr()
    print(type(df_corr_cont))
    print(df_corr_cont)
    sns.heatmap(df_corr_cont)

    # generate correlation metrics for each categorical cont predictor
    corr_matrix_df_list = []
    for pred_outer in cat_predictors:
        corr_matrix_temp_list = []
        for pred_inner in cont_predictors:
            corr_matrix_temp_list.append(
                cc.cat_cont_correlation_ratio(df[pred_outer], df[pred_inner])
            )
        corr_matrix_df_list.append(corr_matrix_temp_list)

    df_corr_matrix = pd.DataFrame(
        corr_matrix_df_list, columns=cont_predictors, index=cat_predictors
    )
    print(df_corr_matrix)
    sns.heatmap(df_corr_matrix, annot=True)
    plt.show()

    # generate correlation metrics for each cat cat predictor
    corr_matrix_df_list = []
    for pred_outer in cat_predictors:
        corr_matrix_temp_list = []
        for pred_inner in cat_predictors:
            corr_matrix_temp_list.append(
                cc.cat_correlation(df[pred_outer], df[pred_inner])
            )
        corr_matrix_df_list.append(corr_matrix_temp_list)

    df_corr_matrix = pd.DataFrame(
        corr_matrix_df_list, columns=cat_predictors, index=cat_predictors
    )
    print(df_corr_matrix)
    sns.heatmap(df_corr_matrix, annot=True)
    plt.show()


if __name__ == "__main__":

    sys.exit(main())
