# Author: Shad Fernandez
# 2021-OCT-11
# BDA 696 HW4

import pandas as pd
import plot_pred_response as prr

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


# calculate t and p values
# regression if continuous
# logistic if categorical
# difference with mean response along  with its plot (check lecture)
# Random forest variable importance ranking for continuous variables
# generate table and all ranking
