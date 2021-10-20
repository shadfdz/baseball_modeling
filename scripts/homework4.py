# Author: Shad Fernandez
# 2021-OCT-11
# BDA 696 HW4

import sys

import bin_response_by_predictor as bin
import pandas as pd
import plot_pred_response as prr
import statsmodels.api as stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 10)


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


def get_linear_model_fit(df, response, pred):
    predictor = stats.add_constant(df[pred])
    lin_reg_model_fit = stats.OLS(df[response], predictor).fit()
    return lin_reg_model_fit


def get_log_model_fit(df, response, pred):
    resp_dummy = pd.get_dummies(df[response])
    resp_dummy_col = resp_dummy.iloc[:, 1]
    predictor = stats.add_constant(df[pred])
    log_reg_model_fit = stats.Logit(resp_dummy_col, predictor).fit()
    return log_reg_model_fit


def get_response_predictors(df):
    print("The following are features of the dataset: ")
    print(*df.columns)
    response_feature = str(
        input("\nPlease enter the column name of the response variable: ")
    )
    resp = [response_feature]
    preds = df.loc[:, ~df.columns.isin(resp)].columns.to_list()
    print("Response: " + resp[0])
    print("Predictors:", *preds)
    return resp, preds


def main():
    # retrieve data
    df = pd.read_csv("../datasets/iris.csv")
    print("The following are features of the dataset: ")
    print(*df.columns)
    response_feature = str(
        input("\nPlease enter the column name of the response variable: ")
    )
    response = [response_feature]
    predictors = df.loc[:, ~df.columns.isin(response)].columns.to_list()
    print("Response: " + response[0])
    print("Predictors:", *predictors)
    response, predictors = get_response_predictors(df)

    print(df.head())
    print(df.columns)

    # loop through predictors and assign if boolean or continuous using get_col_type_dict
    feature_type_dict = get_col_type_dict(df)

    for obj in feature_type_dict.keys():
        print(obj + " " + feature_type_dict.get(obj))

    # create instance of PlotPredictorResponse to generate plots
    feature_plotter = prr.PlotPredictorResponse(df, feature_type_dict)
    # auto generate plot based on data types of response and predictors
    feature_plotter.plot_response_by_predictors(response[0], predictors)

    # generate models and store in dictionary
    model_dict = {}
    for pred in predictors:
        if (
            feature_type_dict.get(response[0]) == "continuous"
            and feature_type_dict.get(pred) == "continuous"
        ):
            model_dict[pred] = get_linear_model_fit(df, response[0], pred)
        elif (
            feature_type_dict.get(response[0]) == "boolean"
            and feature_type_dict.get(pred) == "continuous"
        ):
            model_dict[pred] = get_log_model_fit(df, response[0], pred)
        else:
            print(
                "\nWarning: No model was generated for the feature '"
                + pred
                + "', only continuous predictors are used\n"
            )

    # get summary for each model and print t and p values
    for pred in model_dict.keys():
        model_obj = model_dict.get(pred)
        print("\nRegression Summary for '" + pred + "' as a predictor")
        print(model_obj.summary())
        print("P-value {:.6e}".format(model_obj.tvalues[1]))
        print("T-value {:.6e}\n".format(model_obj.pvalues[1]))

    # Create instance of BinResponseByPredictor to bin response by predictors. Create 10 bins
    bin_response = bin.BinResponseByPredictor(df, feature_type_dict, 10)
    # create data frames of each predictor/response bins and store in dictionary
    bins_dict = bin_response.bin_response_by_predictors(response[0], predictors)

    # Loop through dictionary and print difference with mean of response
    for obj in bins_dict.keys():
        obj_df = bins_dict.get(obj)
        print("Difference with Mean of Response Table")
        print("\tResponse: " + response[0] + ", Predictor: " + obj)
        print(obj_df)
        print(
            "\tDifference with Mean of Response: {:.4e}".format(obj_df["Means"].sum())
        )
        print(
            "\tDifference with Mean of Response Weighted {:.4e}\n".format(
                obj_df["MeanSqrDiffWeighted"].sum()
            )
        )

    # create a random forest model and rank variables by importance
    model_df = df.copy()
    if feature_type_dict.get(response[0]) == "continuous":
        for pred in predictors:
            if feature_type_dict.get(pred) == "boolean":
                label_encoder = LabelEncoder()
                model_df[pred] = label_encoder.fit_transform(df[pred])
        random_forest_model = RandomForestRegressor()
        random_forest_model.fit(model_df[predictors], model_df[response[0]])
        var_importance = random_forest_model.feature_importances_
        for i, score in enumerate(var_importance):
            print("Feature: {}\t\tScore {:.6f}".format(predictors[i], score))
    else:
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(df[predictors], df[response[0]])
        var_importance = random_forest_model.feature_importances_
        for i, score in enumerate(var_importance):
            print("Feature: {}\t\tScore: {:.6f}".format(predictors[i], score))


if __name__ == "__main__":
    sys.exit(main())
