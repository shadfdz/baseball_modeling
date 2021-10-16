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

    response = ["sepal_length"]
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
                print(
                    "/nLinear regression model {} against {}".format(pred, response[0])
                )
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

    df["bin_cat"], bin_array = pd.cut(x=df["sepal_length"], bins=10, retbins=True)
    print(bin_array)
    bin_center = []
    for indx, val in enumerate(bin_array):
        if indx != 0:
            bin_center.append(round((val + bin_array[indx - 1]) / 2, 5))

    bins_cat = df["bin_cat"].sort_values().unique()
    pop_mean = df["sepal_length"].mean()
    data_frame_list = []
    data_frame_columns = ["Bin", "Center", "Counts", "Means", "PopMean", "MeanSqrDiff"]
    for cat, binCenter in zip(bins_cat, bin_center):
        temp_list = []
        temp_list.append(cat)
        temp_list.append(binCenter)
        temp_list.append(df["sepal_length"][(df["bin_cat"] == cat)].count())
        temp_list.append(df["sepal_length"][(df["bin_cat"] == cat)].mean())
        temp_list.append(pop_mean)
        temp_list.append(
            ((df["sepal_length"][(df["bin_cat"] == cat)].mean() - pop_mean) ** 2) / 2
        )
        # pop_prop_list.append(bin_count / len(df.index))
        data_frame_list.append(temp_list)

    bin_df_unweighted = pd.DataFrame(data_frame_list, columns=data_frame_columns)

    print(bin_df_unweighted)

    # data_frame_list = []
    # data_frame_columns = ['Bin','BinCenters','BinCounts','BinMeans(ui)','PopulationMean','MeanSquareDiff',
    # 'PopProportion','MeanSquaredDiffWeighted']
    # for cat, binCenter in zip(bins_cat, bin_center):
    #     temp_list = []
    #     temp_list.append(cat)
    #     temp_list.append(binCenter)
    #     temp_list.append(df['sepal_length'][(df['bin_cat'] == cat)].count())
    #     temp_list.append(df['sepal_length'][(df['bin_cat'] == cat)].mean())
    #     temp_list.append(pop_mean)
    #     temp_list.append(((df['sepal_length'][(df['bin_cat'] == cat)].mean() - pop_mean) ** 2) / 2)
    #     temp_list.append(df['sepal_length'][(df['bin_cat'] == cat)].count() / len(df.index))
    #     temp_list.append(df['sepal_length'][(df['bin_cat'] == cat)].count() / len(df.index))
    #
    #     print(bin_df_unweighted)
    #     data_frame_list.append(temp_list)
    #
    # bin_df_weighted = pd.DataFrame(data_frame_list, columns=data_frame_columns)
    #
    # print(bin_df_weighted)
    #

    # Random forest variable importance ranking for continuous variables
    # generate table and all ranking


if __name__ == "__main__":
    sys.exit(main())
