import sys
from itertools import combinations

import bin_response_by_predictor as brp
import cat_correlation as cc
import html_helper as hh
import pandas as pd
import plot_pred_response as ppr
import pymysql
import statsmodels.api
from plotly import express as px
from scipy import stats
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def set_response_predictors(dataframe):
    """
    Function returns resp name and list of predictors from df
    :param dataframe: data frame
    :return: list of response and predictors
    """
    resp = "win_lose"
    predictors = dataframe.loc[:, ~dataframe.columns.isin([resp])].columns.to_list()
    # # this is where predictors are filtered i.e ID or date time
    # pred_filtered = []
    # for pred in predictors:
    #     pred_filtered.append(pred)
    features = predictors.copy()
    features = features.append(resp)

    return resp, predictors, features


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


def set_cat_cont_predictors(feature_type_dict, resp):
    """
    Functions set predictors to either categorical and continuous lists
    :param feature_type_dict: dict with feature type of each pred
    :param response: response
    :return: feature type dict
    """
    response = resp
    cat_predictors = []
    cont_predictors = []
    for key in feature_type_dict.keys():
        if key != response:
            if feature_type_dict.get(key) == "continuous":
                cont_predictors.append(key)
            else:
                cat_predictors.append(key)
    return cat_predictors, cont_predictors


def create_df_hyperlinks(path_list, name_list, col_name):
    """
    Given a list of file paths and names, function will return df with hyper of name
    with
    :param path_list: list of file paths
    :param name_list: list of names for url link
    :return: df containing hyperlinks of file paths
    """
    temp_table = []
    for path, name in zip(
        path_list,
        name_list,
    ):
        temp_table.append(['<a href="{}">{}</a>'.format(path, name)])

    return pd.DataFrame(temp_table, columns=[col_name], index=name_list)


def to_hyperlinks(path, name):
    """
    Given a list of file paths and names, function will return df with hyper of name
    with
    :param path_list: list of file paths
    :param name_list: list of names for url link
    :return: df containing hyperlinks of file paths
    """
    return '<a href="{}">{}</a>'.format(path, name)


def describe_data_to_html(df, feature_type_dict, resp, pred):
    """
    Function returns a dictionary of dataframes, each of which describes the input df
    Describe includes top 5 and bottom 5 of df, describe, Plots vs response of each pred
    :param df: dataframe
    :param feature_type_dict: a dictionary describing the type of features in the df
    :param resp: (str) response column
    :param pred: (str) list of predictor names
    :return: dict
    """
    # describe data and plot each predictor
    desc_dict = {
        "Top 5 Rows": df.head(5),
        "Bottom 5 Rows": df.tail(5),
        "Data Summary": df.describe().reset_index(),
    }
    # create an html of data description
    desc_html_file = hh.create_html_file(
        "Description",
        "Exploratory Data Analysis",
        desc_dict.keys(),
        desc_dict.values(),
        "describe",
    )
    # Plot Nulls
    plotter = ppr.plot_response_by_predictors(df, feature_type_dict)
    df_null = df.isnull().sum().to_frame(name="null_count")
    plot_file = plotter.plot_simple_bar(
        x=df_null.index.to_list(),
        y=(df_null.null_count / df.shape[0]),
        title="Null Percentage",
        fname="nulls",
    )
    # Append to Description
    hh.append_to_html_file("./output/" + plot_file, desc_html_file)

    return desc_html_file


def show_var_rankings(df, resp, predictor, encode=True):
    """
    Function prints variable rankings
    :param df:
    :param resp:
    :param predictor:
    :param encode:
    :return: pd.Series
    """
    if encode:
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(df[resp])
        df[resp] = label_encoder.transform(df[resp])

    X = df[predictor].values
    y = df[resp].values
    # Random Forest
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X, y)
    # get Variable Ranking
    feature_imp = pd.DataFrame(
        {"Variable": predictor, "Rank": clf.feature_importances_}, index=predictor
    ).sort_values(by="Rank", axis=0, ascending=False)
    return feature_imp


def get_t_p_values(df, resp, predictors):
    """
    :param df:
    :param resp:
    :param predictors:
    :return:
    """
    list_to_df = []
    y = resp
    for pred in predictors:
        row_list = []
        predictor_w_constant = statsmodels.api.add_constant(df[pred])
        linear_regression_model = statsmodels.api.OLS(df[y], predictor_w_constant)

        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {pred}")
        print(linear_regression_model_fitted.summary())

        t_value = round(linear_regression_model_fitted.tvalues[0], 5)
        p_value = round(linear_regression_model_fitted.pvalues[0], 5)
        row_list.append(t_value)
        row_list.append(p_value)

        fig = px.scatter(x=df[pred], y=df[resp], trendline="ols")
        fig.update_layout(
            title=f"Variable: {pred}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {pred}",
            yaxis_title=resp,
        )
        file_path = "scatter_" + resp + "by" + pred + ".html"
        row_list.append(to_hyperlinks(file_path, pred))
        fig.write_html(
            file="./output/" + file_path,
            include_plotlyjs="cdn",
        )
        list_to_df.append(row_list)

    values_df = pd.DataFrame(
        data=list_to_df, columns=["t values", "p values", "plot"], index=predictors
    )
    return values_df


def get_correlation_metrics(df, response, predictors):
    """
    Returns correlation for response and predictor
    :param df: dataframe
    :param response: response variable
    :param predictors: predictor variable
    :return: df of correlation metrics with file links to the plot of each predictor
    """
    corr_list = []
    idx = []
    for pred in combinations(predictors, 2):
        idx.append(pred[0] + pred[1])
        df_corr = df[(df[pred[0]].notnull()) & df[pred[1]].notnull()]
        df_corr = df_corr.reset_index()
        file_link1 = response + "by" + pred[0] + ".html"
        file_link2 = response + "by" + pred[1] + ".html"
        temp_corr_row = [
            '<a href="{}">{}</a>'.format(file_link1, pred[1]),
            '<a href="{}">{}</a>'.format(file_link2, pred[0]),
            stats.pearsonr(df_corr[pred[1]], df_corr[pred[0]])[0],
        ]
        corr_list.append(temp_corr_row)
    col_names = ["Pred1", "Pred2", "Correlation"]
    corr_list_df = pd.DataFrame(corr_list, columns=col_names, index=idx)
    return corr_list_df


def get_correlation(df, pred1, pred2, feature_type_dict):
    """
    Function gets correlation based on feature type dict
    :param df: dataframe
    :param pred1: predictor variable 1
    :param pred2: predictor variable 2
    :param feature_type_dict: dictionary containing feature data type
    :return: correlation value
    """

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


def get_correlation_matrix(df, list1_pred, list2_pred, feature_type_dict):
    """
    Creates a correlation matrix between two features
    :param df: dataframe
    :param list1_pred: list of predictors
    :param list2_pred: list of predictors
    :param feature_type_dict: dictionary containing feature types
    :return: correlation matrix as a dataframe
    """

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


def get_df_as_matrix(
    df, list_pred1, list_pred2, category, attribute, midpoint1=False, midpoint2=False
):
    """
    Pivots a df into a matrix with correlation as intervals in each dimension . Midpoint true shows interval center
    :param df: dataframe
    :param list_pred1: columns categories
    :param list_pred2: columns categories
    :param category: feature
    :param attribute:
    :param midpoint1: bin mid point
    :param midpoint2: bin midpoint
    :return: dataframe of correlation matrix
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


def get_brute_force_tables(df, resp, predictors, feature_type_dict):
    """
    :param df: dataframe
    :param resp: response
    :param predictors: predictors
    :param feature_type_dict: feature type dict
    :return:
    """
    response_bins = brp.BinResponseByPredictor(df, feature_type_dict)

    # brute force table for predictors
    brute_force_table_df = []
    idx = []
    for pred in combinations(predictors, 2):
        idx.append(pred[0] + pred[1])
        title = pred[0] + "And" + pred[1]
        df_bins, bin_list1, bin_list2 = response_bins.bin_2d_response_by_predictors(
            resp, pred[0], pred[1], 8
        )
        int_1 = get_bin_intervals(bin_list1)
        int_2 = get_bin_intervals(bin_list2)
        file_name = pred[0] + pred[1] + "response"
        file_link = ppr.plot_correlation_matrix(
            get_df_as_matrix(df_bins, int_1, int_2, "Bin", "RespBinMean", True, True),
            title,
            file_name,
        )
        brute_force_table_df.append(
            [pred[0], pred[1], to_hyperlinks(file_link, pred[0] + " and " + pred[1])]
        )

    brute_force_html_df = pd.DataFrame(
        brute_force_table_df,
        columns=["Pred1", "Pred2", "Combined Response Plot"],
        index=idx,
    )
    return brute_force_html_df


def main():

    # get data
    connection = pymysql.connect(host="db", user="", password="", db="baseball")

    cursor = connection.cursor()
    query = "Select * from baseball_stats;"
    df_raw_data = pd.read_sql(query, connection)
    cursor.close()

    # print raw data nulls
    print(df_raw_data.isnull().sum())

    df = df_raw_data.drop(["local_date", "Year"], axis=1)

    # set resp and pred and get cat and cont predictors
    resp, pred, features = set_response_predictors(df)
    feature_type_dict = set_feature_type_dict(df)
    cat_pred, cont_pred = set_cat_cont_predictors(feature_type_dict, resp)

    # create dictionary of dataframes for html conversion
    # get description of data in df
    to_html_dict = {
        "Top 5 Rows": df.head(5),
        "Bottom 5 Rows": df.tail(5),
        "Data Summary": df.describe(),
    }

    # plot predictors
    plot_path_list = ppr.plot_response_by_predictors(df, resp, pred, feature_type_dict)
    pred_plot_df = create_df_hyperlinks(plot_path_list, pred, "Plots")

    # plot binned mean response
    bin_response = brp.BinResponseByPredictor(df, feature_type_dict, 8)
    bin_response_df_list = bin_response.bin_response_by_predictors(resp, pred)
    bin_plot_paths = []
    for bins, predictor in zip(bin_response_df_list, pred):
        print(predictor)
        print(bins)
        bin_plot_paths.append(ppr.plot_diff_with_mor(bins, resp, predictor))
    bin_plots_df = create_df_hyperlinks(bin_plot_paths, pred, "Mean of Response")
    pred_plot_df = pred_plot_df.join(bin_plots_df)

    # get variable ranking table
    var_ranking_df = show_var_rankings(df, resp, pred)
    pred_plot_df = pred_plot_df.join(var_ranking_df)

    # get t and p values and plot
    t_p_df = get_t_p_values(df, resp, pred)
    pred_plot_df = pred_plot_df.join(t_p_df)

    move_to_first = pred_plot_df.pop("Variable")
    pred_plot_df.insert(0, "Variable", move_to_first)
    pred_plot_df = pred_plot_df.sort_values(by="Rank", ascending=False)
    to_html_dict["Predictors Plots and Metrics"] = pred_plot_df

    # generate correlation matrix for predictors
    corr_matrix = get_correlation_matrix(df, cont_pred, cont_pred, feature_type_dict)
    corr_plot_path_list = ppr.plot_correlation_matrix(
        corr_matrix, "Continuous", "cont_cont_pred"
    )
    corr_plot_df = create_df_hyperlinks(
        [corr_plot_path_list], ["Correlation Matrix"], "Plots"
    )
    to_html_dict["Correlation Matrix"] = corr_plot_df

    # Get Correlation Metrics between predictors
    corr_metrics_df = get_correlation_metrics(df, resp, pred)

    # Brute force Tables
    brute_force_tables_df = get_brute_force_tables(df, resp, pred, feature_type_dict)

    # combine brute force and correlation metrics
    corr_metrics_df = corr_metrics_df.merge(
        brute_force_tables_df["Combined Response Plot"],
        right_index=True,
        left_index=True,
    )
    corr_metrics_df = corr_metrics_df.sort_values(by="Correlation", ascending=False)
    to_html_dict["Brute Force and Correlation Metrics"] = corr_metrics_df

    # droppings some columns that are highly correlated
    df_processed = df.drop(
        axis=1,
        columns=[
            "SP_WHIP_d",
            "SP_FIP_d",
            "SP_tb_pt",
            "TP_hr_so",
            "TP_sdt_fo",
            "TP_a_so_10",
            "TB_ARHR",
            "TB_BA",
        ],
    )

    # create new feature type dict without dropped columns and new features
    feature_type_dict_processed = set_feature_type_dict(df_processed)
    resp, pred, features = set_response_predictors(df_processed)
    cat_predictors_proc, cont_predictors_proc = set_cat_cont_predictors(
        feature_type_dict_processed, resp
    )

    # scale data
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_processed[cont_predictors_proc])
    df_processed_model = pd.DataFrame(scaled_features, columns=cont_predictors_proc)
    df_processed_model["win_lose"] = df_processed["win_lose"]
    df_processed_model["Year"] = df_raw_data["Year"]

    # Split data with train as data in 2011
    X_train = df_processed_model[cont_predictors_proc][
        df_processed_model["Year"] < 2011
    ].values
    X_test = df_processed_model[cont_predictors_proc][
        df_processed_model["Year"] == 2011
    ].values
    y_train = (
        df_processed_model["win_lose"][df_processed_model["Year"] < 2011]
        .drop(axis=1, columns=["Year"])
        .values
    )
    y_test = (
        df_processed_model["win_lose"][df_processed_model["Year"] == 2011]
        .drop(axis=1, columns=["Year"])
        .values
    )

    # Create List of Models and Metrics
    model_list = ["Logistic Regression", "Random Forest", "LDA", "SVM"]
    labels = ["lose", "win"]
    accuracy_list = []
    auc_list = []
    tpr_list = []
    fpr_list = []

    # Log Regression
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    log_model.fit(X_train, y_train)

    y_pred = log_model.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))
    # roc https: // www.statology.org / plot - roc - curve - python /
    y_pred_prob_log = log_model.predict_proba(X_test)[::, 1]
    auc_list.append(roc_auc_score(y_test, y_pred_prob_log))
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_log)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    # create confusion matrix
    log_model_cm = ppr.plot_confusion_matrix(
        y_test, y_pred, labels, "Logistic Regression"
    )

    # Random Forest
    # https: // machinelearningmastery.com / random - forest - ensemble - in -python /
    rf_model = RandomForestClassifier(random_state=1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))
    # get auc and tpr and fpr for roc plot
    y_pred_prob_rf = rf_model.predict_proba(X_test)[::, 1]
    auc_list.append(roc_auc_score(y_test, y_pred_prob_rf))
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_rf)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    # create confusion matrix
    random_forest_cm = ppr.plot_confusion_matrix(
        y_test, y_pred, labels, "Random Forest"
    )

    # LDA
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)
    y_pred = lda_model.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))
    y_pred_prob_lda = lda_model.predict_proba(X_test)[::, 1]
    auc_list.append(roc_auc_score(y_test, y_pred_prob_lda))
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_lda)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    # create confusion matrix
    lda_cm = ppr.plot_confusion_matrix(y_test, y_pred, labels, "LDA")

    # SVM
    svm_model = svm.SVC(probability=True)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))
    y_pred_prob_svm = svm_model.predict_proba(X_test)[::, 1]
    auc_list.append(roc_auc_score(y_test, y_pred_prob_svm))
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_svm)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    # create confusion matrix
    svm_cm = ppr.plot_confusion_matrix(y_test, y_pred, labels, "SVM")

    result_df = pd.DataFrame(
        {"Model": model_list, "Accuracy": accuracy_list, "AUC": auc_list}
    )

    to_html_dict["Results"] = result_df

    hh.create_html_file(
        "index",
        "Baseball Data Set",
        to_html_dict.keys(),
        to_html_dict.values(),
        "describe",
    )

    # append confusion matrices
    hh.append_to_html_file(log_model_cm, "./output/index.html")
    hh.append_to_html_file(random_forest_cm, "./output/index.html")
    hh.append_to_html_file(lda_cm, "./output/index.html")
    hh.append_to_html_file(svm_cm, "./output/index.html")

    # append roc plot to index
    roc_plot_file = ppr.plot_roc(model_list, fpr_list, tpr_list, auc_list)
    hh.append_to_html_file(roc_plot_file, "./output/index.html")

    # pickl the best model


if __name__ == "__main__":
    sys.exit(main())
