# Author: Shad Fernandez
# 2021-OCT-11
# BDA 696 HW4

import sys

import pandas as pd
import plot_pred_response as prr
import statsmodels.api as stats


def main():
    df = pd.read_csv("../datasets/Iris.csv")
    print(df.dtypes)

    response = ["Class"]
    predictors = ["sepal_width", "petal_length", "petal_width"]

    # loop through predictors
    # check using feature data type classifier
    feature_type_dict = {}
    for col in df.columns:
        # may change to like 'int' since could be int32
        if df[col].dtype == "int64":
            if df[col].unique().size == 2:
                feature_type_dict[col] = "categorical"
            else:
                feature_type_dict[col] = "continuous"
        elif df[col].dtype == "float64":
            feature_type_dict[col] = "continuous"
        else:
            feature_type_dict[col] = "categorical"

    # generate necessary plots
    feature_plotter = prr.PlotPredictorResponse(df, feature_type_dict)
    for pred in predictors:
        if feature_type_dict.get(response[0]) == "continuous":
            if feature_type_dict.get(pred) == "continuous":
                feature_plotter.cont_resp_cont_pred(response[0], pred)
            else:
                feature_plotter.cont_resp_cat_pred(response[0], pred)
        else:
            if feature_type_dict.get(pred) == "categorical":
                feature_plotter.cat_resp_cat_pred(response[0], pred)
            else:
                feature_plotter.cat_resp_cont_pred(response[0], pred)

    if feature_type_dict.get(response[0]) == "continuous":
        for pred in predictors:
            if feature_type_dict.get(pred) == "continuous":
                predictor = stats.add_constant(df[pred])
                lin_reg_model_fit = stats.OLS(df[response[0]], predictor).fit()
                print("Linear regression model {} against {}".format(pred, response[0]))
                print(lin_reg_model_fit.summary())
            else:
                print("Please include at least one continuous predictor")
    else:
        # convert response to dummy code dummy!
        # when error is thrown catch near zero variance
        resp_dummy = pd.get_dummies(df[response[0]])
        resp_dummy_df = resp_dummy.iloc[:, 1]
        for pred in predictors:
            if feature_type_dict.get(pred) == "continuous":
                predictor = stats.add_constant(df[pred])
                log_reg_model_fit = stats.Logit(resp_dummy_df, predictor).fit()
                print("Log regression model {} against {}".format(pred, response[0]))
                print(log_reg_model_fit.summary())
            else:
                print("Please include at least one continuous predictor")

    # difference with mean response along  with its plot (check lecture)
    # Random forest variable importance ranking for continuous variables
    # generate table and all ranking


if __name__ == "__main__":
    sys.exit(main())
