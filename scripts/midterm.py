import sys

import cat_correlation as cc
import pandas as pd
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


def get_predictor_type_dict(dataframe):
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
    predictor_type_dict = get_predictor_type_dict(df)
    cat_predictors = []
    cont_predictors = []
    for key in predictor_type_dict.keys():
        if predictor_type_dict.get(key) == "continuous":
            cont_predictors.append(key)
        else:
            cat_predictors.append(key)
    print(cat_predictors)
    print(cont_predictors)

    # # print correlation metrics for each continuous predictor
    for pred_outer in cont_predictors:
        for pred_inner in cont_predictors:
            if pred_outer != pred_inner:
                df_corr = df[(df[pred_inner].notnull()) & df[pred_outer].notnull()]
                print(stats.pearsonr(df_corr[pred_outer], df_corr[pred_inner]))

    # print correlation metrics for each categorical cont predictor
    for pred_outer in cat_predictors:
        for pred_inner in cat_predictors:
            if pred_outer != pred_inner:
                print(
                    pred_outer,
                    pred_inner,
                    cc.cat_correlation(df[pred_outer], df[pred_inner]),
                )

    # print correlation metrics for each categorical predictor
    for pred_outer in cat_predictors:
        for pred_inner in cont_predictors:
            if pred_outer != pred_inner:
                print(
                    pred_outer,
                    pred_inner,
                    cc.cat_cont_correlation_ratio(df[pred_outer], df[pred_inner]),
                )

    # print in table by metrics in descending order

    # put links to original variable plot done in hw4

    # generate correlation matrices for the above three

    # print heat map of all three combinations


if __name__ == "__main__":

    sys.exit(main())
