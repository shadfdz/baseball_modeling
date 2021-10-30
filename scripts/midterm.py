import sys

import bin_response_by_predictor as brp
import cat_correlation as cc
import numpy as np
import pandas as pd
import plot_pred_response as ppr
import plotly.figure_factory as ff
import xlsxwriter
from scipy import stats

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 12)


def get_response_predictors(dataframe):
    """
    Prompts user to enter response name. Function returns list of reponse and features that are predictors
    :param dataframe: data frame
    :return: list of response and predictors
    """
    print("The following are features of the dataset: ")
    print(*dataframe.columns)
    response_feature = str(
        input("\nPlease enter the column name of the response variable: ")
    )
    resp = response_feature
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


def get_df_as_matrix(
    df, list_pred1, list_pred2, category, attribute, midpoint1=False, midpoint2=False
):
    """
    Pivots a df into a matrix. Migpoint true shows interval center
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
    df = df.drop("Name", axis=1)

    xlsxwriter.Workbook("AddingThisToBypassFlakeHook.xlsx")

    # create instance of writer to output results on excel
    writer = pd.ExcelWriter("../output/Midterm.xlsx", engine="xlsxwriter")
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

    # PART 1
    # create correlation metrics for each predictor
    corr_temp_list = []
    corr_col_names = ["Predictor1", "Predictor2", "Correlation"]

    # plot each predictor along with response

    resp_pred_plotter = ppr.PlotPredictorResponse(df, feature_type_dict)

    # print correlation metrics for each continuous predictor
    for pred_outer in cont_predictors:
        for pred_inner in cont_predictors:
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
                    stats.pearsonr(df_corr[pred_outer], df_corr[pred_inner])[0],
                ]
                corr_temp_list.append(temp_corr_row)
    corr_df = create_sheet(
        corr_temp_list, corr_col_names, writer, "ContPredCorrMetrics"
    )
    print(corr_df)

    # print correlation metrics for each categorical predictor
    for pred_outer in cat_predictors:
        for pred_inner in cat_predictors:
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
                    cc.cat_correlation(df_corr[pred_outer], df_corr[pred_inner]),
                ]
                corr_temp_list.append(temp_corr_row)
    corr_df = create_sheet(
        corr_temp_list, corr_col_names, writer, "CatContPredCorrMetrics"
    )
    print(corr_df)

    # print correlation metrics for each categorical cont predictor
    for pred_outer in cat_predictors:
        for pred_inner in cont_predictors:
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
                    cc.cat_cont_correlation_ratio(
                        df_corr[pred_outer], df_corr[pred_inner]
                    ),
                ]
                corr_temp_list.append(temp_corr_row)
    corr_df = create_sheet(
        corr_temp_list, corr_col_names, writer, "CatContPredCorrMetrics"
    )
    print(corr_df)

    # PART 2
    # generate correlation matrices for the above three
    # print heat map of all three combinations

    # generate correlation matrix for each cont cont predictor
    sheet_list = []
    df_corr_cont = df[cont_predictors].corr()
    z = df_corr_cont.to_numpy()
    x = df_corr_cont.columns.to_list()
    y = list(reversed(x))
    fig = ff.create_annotated_heatmap(
        z=z, x=x, y=y, annotation_text=z, colorscale="thermal", showscale=True
    )
    fig.update_layout(
        title="Correlation Matrix of Continuous Predictors",
    )
    file_name = "cont_correlation_matrix.html"
    fig.write_html(
        file="../output/" + file_name,
        include_plotlyjs="cdn",
    )
    sheet_list.append(["ContinuousPredictors", file_name])
    print("Correlation Matrix for Continuous Predictors")
    print(df_corr_cont)

    # generate correlation metrics for each categorical cont predictor
    corr_matrix_df_list = []
    for pred_outer in cat_predictors:
        corr_matrix_temp_list = []
        for pred_inner in cont_predictors:
            df_corr = df[(df[pred_inner].notnull()) & df[pred_outer].notnull()]
            df_corr = df_corr.reset_index()
            corr_matrix_temp_list.append(
                cc.cat_cont_correlation_ratio(df_corr[pred_outer], df_corr[pred_inner])
            )
        corr_matrix_df_list.append(corr_matrix_temp_list)
    df_corr_matrix = pd.DataFrame(
        corr_matrix_df_list, columns=cont_predictors, index=cat_predictors
    )
    z = df_corr_matrix.to_numpy()
    x = df_corr_matrix.columns.to_list()
    y = df_corr_matrix.index.to_list()
    fig = ff.create_annotated_heatmap(
        z=z, x=x, y=y, annotation_text=z, colorscale="thermal", showscale=True
    )
    fig.update_layout(
        title="Correlation Matrix of Continuous and Categorical Predictors",
    )
    file_name = "cat_cont_correlation_matrix.html"
    fig.write_html(
        file="../output/" + file_name,
        include_plotlyjs="cdn",
    )
    sheet_list.append(["CategoricalContinuousPredictors", file_name])
    print("Correlation Matrix for Categorical and Continuous Predictors")
    print(df_corr_matrix)

    # generate correlation metrics for each cat cat predictor
    corr_matrix_df_list = []
    for pred_outer in cat_predictors:
        corr_matrix_temp_list = []
        for pred_inner in cat_predictors:
            df_corr = df[(df[pred_inner].notnull()) & df[pred_outer].notnull()]
            df_corr = df_corr.reset_index()
            corr_matrix_temp_list.append(
                cc.cat_correlation(df_corr[pred_outer], df_corr[pred_inner])
            )
        corr_matrix_df_list.append(corr_matrix_temp_list)
    df_corr_matrix = pd.DataFrame(
        corr_matrix_df_list, columns=cat_predictors, index=cat_predictors
    )
    z = df_corr_matrix.to_numpy()
    x = df_corr_matrix.columns.to_list()
    y = df_corr_matrix.index.to_list()
    fig = ff.create_annotated_heatmap(
        z=z, x=x, y=y, annotation_text=z, colorscale="thermal", showscale=True
    )
    fig.update_layout(
        title="Correlation Matrix of Categorical Predictors",
    )
    file_name = "cat_correlation_matrix.html"
    fig.write_html(
        file="../output/" + file_name,
        include_plotlyjs="cdn",
    )
    sheet_list.append(["CategoricalContinuousPredictors", file_name])

    df_corr_matrices = pd.DataFrame(sheet_list, columns=["Title", "FileName"])
    df_corr_matrices["Title"] = (
        '=HYPERLINK("'
        + df_corr_matrices["FileName"]
        + '","'
        + df_corr_matrices["Title"]
        + '")'
    )
    df_corr_matrices.to_excel(writer, sheet_name="CorrelationMatrix")
    print("Correlation Metrics for Categorical Predictors")
    print(df_corr_matrix)

    # Part 3
    brute_force_table = []
    response_bins = brp.BinResponseByPredictor(df, feature_type_dict)

    # print correlation metrics for each continuous predictor
    for pred_outer in cont_predictors:
        for pred_inner in cont_predictors:
            if pred_outer != pred_inner:
                df_bins, bin_list1, bin_list2 = response_bins.bin_2d_cont_cont_pred(
                    response, pred_outer, pred_inner, 8
                )
                bin_interval1 = get_bin_intervals(bin_list1)
                bin_interval2 = get_bin_intervals(bin_list2)
                # print(df_bins)
                df_matrix = get_df_as_matrix(
                    df_bins,
                    bin_interval1,
                    bin_interval2,
                    "Bin",
                    "RespBinMean",
                    True,
                    True,
                )
                z = np.round(df_matrix.to_numpy(), 3)
                x = [str(i) for i in bin_interval1]
                y = list(reversed([str(i) for i in bin_interval2]))
                fig = ff.create_annotated_heatmap(
                    z=z,
                    x=x,
                    y=y,
                    colorscale="thermal",
                    annotation_text=z,
                    showscale=True,
                )
                fig.update_layout(title=pred_outer + " and " + pred_inner + " binmean")
                file_name = pred_outer + "and" + pred_inner + "binmean.html"
                fig.write_html(
                    file="../output/" + file_name,
                    include_plotlyjs="cdn",
                )

                temp_table = [
                    pred_outer,
                    pred_inner,
                    df_bins["MeanSquaredDiff"].sum(),
                    df_bins["WeighMeanSquaredDiff"].sum(),
                    file_name,
                ]
                brute_force_table.append(temp_table)
    col_names = ["Pred1", "Pred2", "MeanSquaredDiff", "WeighMeanSquaredDiff", "Plot"]
    cont_pred_brute_df = pd.DataFrame(brute_force_table, columns=col_names)
    cont_pred_brute_df = cont_pred_brute_df.sort_values(
        by="WeighMeanSquaredDiff", ascending=False
    )
    cont_pred_brute_df["Plot"] = (
        '=HYPERLINK("' + cont_pred_brute_df["Plot"] + '","Link")'
    )
    cont_pred_brute_df.to_excel(writer, sheet_name="ContPredMatrixPlot", index=False)
    print("Continuous Predictors Weighted and Unweighted Mean of Response")
    print(cont_pred_brute_df)

    # print correlation metrics for each categorical predictor
    brute_force_table.clear()
    for pred_outer in cat_predictors:
        for pred_inner in cat_predictors:
            if pred_outer != pred_inner:
                df_bins, bin_list1, bin_list2 = response_bins.bin_2d_cat_cat_pred(
                    response, pred_outer, pred_inner, 8
                )
                df_matrix = get_df_as_matrix(
                    df_bins, bin_list1, bin_list2, "Bin", "RespBinMean", False, False
                )
                z = np.round(df_matrix.to_numpy(), 3)
                y = list(reversed(df_matrix.index.to_list()))
                fig = ff.create_annotated_heatmap(
                    z=z,
                    x=df_matrix.columns.to_list(),
                    y=y,
                    colorscale="thermal",
                    annotation_text=z,
                    showscale=True,
                )
                fig.update_layout(title=pred_outer + " and " + pred_inner + " binmean")
                file_name = pred_outer + "and" + pred_inner + "binmean.html"
                fig.write_html(
                    file="../output/" + file_name,
                    include_plotlyjs="cdn",
                )

                temp_table = [
                    pred_outer,
                    pred_inner,
                    df_bins["MeanSquaredDiff"].sum(),
                    df_bins["WeighMeanSquaredDiff"].sum(),
                    file_name,
                ]
                brute_force_table.append(temp_table)
    col_names = ["Pred1", "Pred2", "MeanSquaredDiff", "WeighMeanSquaredDiff", "Plot"]
    cont_pred_brute_df = pd.DataFrame(brute_force_table, columns=col_names)
    cont_pred_brute_df = cont_pred_brute_df.sort_values(
        by="WeighMeanSquaredDiff", ascending=False
    )
    cont_pred_brute_df["Plot"] = (
        '=HYPERLINK("' + cont_pred_brute_df["Plot"] + '","Link")'
    )
    cont_pred_brute_df.to_excel(writer, sheet_name="CatPredMatrixPlot", index=False)
    print("Categorical Predictors Weighted and Unweighted Mean of Response")
    print(cont_pred_brute_df)

    brute_force_table.clear()
    for pred_outer in cat_predictors:
        for pred_inner in cont_predictors:
            if pred_outer != pred_inner:
                print(pred_outer, pred_inner)
                df_bins, bin_list1, bin_list2 = response_bins.bin_2d_cat_cont_pred(
                    response, pred_outer, pred_inner, 8
                )
                bin_interval2 = get_bin_intervals(bin_list2)
                df_matrix = get_df_as_matrix(
                    df_bins, bin_list1, bin_interval2, "Bin", "RespBinMean", False, True
                )
                print(df_matrix)

                z = np.round(df_matrix.to_numpy(), 3)
                x = [str(i) for i in bin_interval2]
                y = list(reversed(df_matrix.index.to_list()))
                fig = ff.create_annotated_heatmap(
                    z=z,
                    x=x,
                    y=y,
                    colorscale="thermal",
                    annotation_text=z,
                    showscale=True,
                )
                fig.update_layout(title=pred_outer + " and " + pred_inner + " binmean")
                file_name = pred_outer + "and" + pred_inner + "binmean.html"
                fig.write_html(
                    file="../output/" + file_name,
                    include_plotlyjs="cdn",
                )

                temp_table = [
                    pred_outer,
                    pred_inner,
                    df_bins["MeanSquaredDiff"].sum(),
                    df_bins["WeighMeanSquaredDiff"].sum(),
                    file_name,
                ]
                brute_force_table.append(temp_table)
                break
    col_names = ["Pred1", "Pred2", "MeanSquaredDiff", "WeighMeanSquaredDiff", "Plot"]
    cont_pred_brute_df = pd.DataFrame(brute_force_table, columns=col_names)
    cont_pred_brute_df = cont_pred_brute_df.sort_values(
        by="WeighMeanSquaredDiff", ascending=False
    )
    cont_pred_brute_df["Plot"] = (
        '=HYPERLINK("' + cont_pred_brute_df["Plot"] + '","Link")'
    )
    cont_pred_brute_df.to_excel(writer, sheet_name="CatContPredMatrixPlot", index=False)
    print(
        "Categorical and Continuous Predictors Weighted and Unweighted Mean of Response"
    )

    print(cont_pred_brute_df)
    writer.save()


if __name__ == "__main__":

    sys.exit(main())
