# Author: Shad Fernandez
# 2021-OCT-11
# BDA 696 HW4

import pandas as pd

df = pd.read_csv("../datasets/iris.csv")

# assign predictors and response
# predictors = df.columns[[0, 1, 2, 4]]
# response = df.columns[3]
# data_type = {}
print(df.dtypes)
# print(test)

response = ["sepal_length"]
predictors = ["sepal_width", "petal_length", "petal_width", "Class"]

# loop through predictors
# check using feature data type classifier
pred_col_dtype_dict = {}
for col in df.columns:
    # may change to like 'int' since could be int32
    if df[col].dtype == "int64":
        if df[col].unique().size == 2:
            pred_col_dtype_dict[col] = "categorical"
        else:
            pred_col_dtype_dict[col] = "continuous"
    elif df[col].dtype == "float64":
        pred_col_dtype_dict[col] = "continuous"
    else:
        pred_col_dtype_dict[col] = "categorical"

for val in pred_col_dtype_dict.keys():
    print(val + " " + pred_col_dtype_dict.get(val))

# generate necessary plots
# loop through predictor list
# check dictionary
# if cat response -> pred boolean - heat plot
# if cat response -> pred continous - violin plot on pred grouped by response or distr


# calculate t and p values
# regression if continuous
# logistic if categorical
# difference with mean response along  with its plot (check lecture)
# Random forest variable importance ranking for continuous variables
# generate table and all ranking
