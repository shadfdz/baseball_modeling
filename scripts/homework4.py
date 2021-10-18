# Author: Shad Fernandez
# 2021-OCT-11
# BDA 696 HW4

import sys

import binresponsebypredictor as bin
import pandas as pd
import plot_pred_response as prr
import statsmodels.api as stats


def get_col_type_dict(dataframe):
    """

    :param dataframe:
    :return:
    """
    feature_type_dict = {}
    for col in dataframe.columns:
        if dataframe[col].dtype == "int64":
            if dataframe[col].unique().size == 2:
                feature_type_dict[col] = "boolean"
            else:
                feature_type_dict[col] = "continuous"
        elif dataframe[col].dtype == "float64":
            feature_type_dict[col] = "continuous"
        else:
            feature_type_dict[col] = "boolean"
    return feature_type_dict


def main():
    df = pd.read_csv("../datasets/Iris.csv")
    print(df.columns)
    # response_col = str(input("Please enter the column name of the response variable: "))
    response = ["petal_width"]
    predictors = df.loc[:, ~df.columns.isin(response)].columns.to_list()
    print("Response: " + response[0])
    print("Predictors:")
    print(predictors)

    # loop through predictors and assign if boolean or continuous
    feature_type_dict = get_col_type_dict(df)

    # generate instance of PlotPredictorResponse
    feature_plotter = prr.PlotPredictorResponse(df, feature_type_dict)
    # # auto generate plot based on response data type
    # feature_plotter.plot_auto(response, predictors)

    # create models and fit according to dtype
    if feature_type_dict.get(response[0]) == "continuous":
        for pred in predictors:
            if feature_type_dict.get(pred) == "continuous":
                predictor = stats.add_constant(df[pred])
                lin_reg_model_fit = stats.OLS(df[response[0]], predictor).fit()
                print(
                    "/nLinear regression model {} against {}".format(pred, response[0])
                )
                print(lin_reg_model_fit.summary())
            else:
                print("Please include at least one continuous predictor")
    else:
        resp_dummy = pd.get_dummies(df[response[0]])
        resp_dummy_df = resp_dummy.iloc[:, 1]
        for pred in predictors:
            if feature_type_dict.get(pred) == "continuous":
                predictor = stats.add_constant(df[pred])
                log_reg_model_fit = stats.Logit(resp_dummy_df, predictor).fit()
                print("Log regression model {} against {}".format(pred, response[0]))
                print(log_reg_model_fit.summary())
            else:
                # change this logic
                print("cccheee Please include at least one continuous predictor")

    # Difference with Mean of Response vs Bin for Each predictor
    bin_response = bin.BinResponseByPredictor(df)
    if feature_type_dict.get(response[0]) == "continuous":
        for pred in predictors:
            if feature_type_dict.get(pred) == "continuous":
                df_bin = bin_response.bin_cont_resp_cont_pred(response[0], pred)
                print("Difference with Mean of Response Unweighted")
                print(
                    df_bin.loc[
                        :,
                        ~df_bin.columns.isin(["PopProportion", "MeanSqrDiffWeighted"]),
                    ]
                )
                print("Difference with Mean of Response Weighted")
                print(df_bin)
                feature_plotter.plot_diff_with_MOR(df_bin, response[0], pred)
            else:
                df_bin = bin_response.bin_cont_resp_cat_pred(response[0], pred)
                print("Difference with Mean of Response Unweighted")
                print(
                    df_bin.loc[
                        :,
                        ~df_bin.columns.isin(["PopProportion", "MeanSqrDiffWeighted"]),
                    ]
                )
                print("Difference with Mean of Response Weighted")
                print(df_bin)
                feature_plotter.plot_diff_with_MOR(df_bin, response[0], pred)
    else:
        for pred in predictors:
            if feature_type_dict.get(pred) == "boolean":
                df_bin = bin_response.bin_cat_resp_cat_pred(response[0], pred)
                print("Difference with Mean of Response Unweighted")
                print(
                    df_bin.loc[
                        :,
                        ~df_bin.columns.isin(["PopProportion", "MeanSqrDiffWeighted"]),
                    ]
                )
                print("Difference with Mean of Response Weighted")
                print(df_bin)
                feature_plotter.plot_diff_with_MOR(df_bin, response[0], pred)
            else:
                df_bin = bin_response.bin_cat_resp_cont_pred(response[0], pred)
                print("Difference with Mean of Response Unweighted")
                print(
                    df_bin.loc[
                        :,
                        ~df_bin.columns.isin(["PopProportion", "MeanSqrDiffWeighted"]),
                    ]
                )
                print("Difference with Mean of Response Weighted")
                print(df_bin)
                feature_plotter.plot_diff_with_MOR(df_bin, response[0], pred)

    # Random forest variable importance ranking for continuous variables
    # generate table and all ranking


if __name__ == "__main__":
    sys.exit(main())
