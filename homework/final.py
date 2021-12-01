import sys

import html_helper as hh
import pandas as pd
import plot_pred_response as ppr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# check for skewnes?!
def set_response_predictors(dataframe):
    """
    Function returns resp name and list of predictors from df
    :param dataframe: data frame
    :return: list of response and predictors
    """
    resp = "Class"
    predictors = dataframe.loc[:, ~dataframe.columns.isin([resp])].columns.to_list()
    pred_filtered = []
    for pred in predictors:
        pred_filtered.append(pred)
    features = pred_filtered.copy()
    features.append(resp)

    return resp, pred_filtered, features


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


def set_cat_cont_predictors(feature_type_dict, response):
    """
    Functions set predictors to either categorical and continuous lists
    :param feature_type_dict: dict with feature type of each pred
    :param response: response
    :return: feature type dict
    """
    cat_predictors = []
    cont_predictors = []
    for key in feature_type_dict.keys():
        if key != response:
            if feature_type_dict.get(key) == "continuous":
                cont_predictors.append(key)
            else:
                cat_predictors.append(key)
    return cat_predictors, cont_predictors


def create_df_hyperlinks(path_list, name_list):
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

    return pd.DataFrame(temp_table, columns=["Predictors"], index=name_list)


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
    # plot predictors
    plotter = ppr.PlotPredictorResponse(df, feature_type_dict)
    plot_path_list = plotter.plot_response_by_predictors(resp, pred)
    pred_plot_df = create_df_hyperlinks(plot_path_list, pred)
    # describe data and plot each predictor
    desc_dict = {
        "Top 5 Rows": df.head(5),
        "Bottom 5 Rows": df.tail(5),
        "Data Summary": df.describe().reset_index(),
        "Predictor vs Response Plot and Variable Rankings": pred_plot_df.join(
            show_var_rankings(df, resp, pred)
        )
        .sort_values(by="Rank", axis=0, ascending=False)
        .drop(columns=["Variable"]),
    }
    return desc_dict


def show_var_rankings(df, resp, pred, encode=True):
    """
    Function prints variable rankings
    :param df:
    :param resp:
    :param pred:
    :param encode:
    :return: pd.Series
    """
    if encode:
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(df[resp])
        df[resp] = label_encoder.transform(df[resp])

    X = df[pred].values
    y = df[resp].values
    # Random Forest
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X, y)
    # get Variable Ranking
    feature_imp = pd.DataFrame(
        {"Variable": pred, "Rank": clf.feature_importances_}, index=pred
    ).sort_values(by="Rank", axis=0, ascending=False)
    return feature_imp


def main():

    # get data
    df_raw = pd.read_csv("../datasets/Iris.csv")

    # set resp and pred and get cat and cont predictors
    resp, pred, features = set_response_predictors(df_raw)
    df = df_raw[features]
    feature_type_dict = set_feature_type_dict(df)
    cat_pred, cont_pred = set_cat_cont_predictors(feature_type_dict, resp)

    # PART 1 EDA
    # get description of data in df
    desc_dict = describe_data_to_html(df, feature_type_dict, resp, pred)

    # create an html of data description
    desc_html_file = hh.create_html_file(
        "Description",
        "Exploratory Data Analysis",
        desc_dict.keys(),
        desc_dict.values(),
        "describe",
    )

    # Plot Nulls
    plotter = ppr.PlotPredictorResponse(df, feature_type_dict)
    df_null = df.isnull().sum().to_frame(name="null_count")
    plot_file = plotter.plot_simple_bar(
        x=df_null.index.to_list(),
        y=(df_null.null_count / df.shape[0]),
        title="Null Percentage",
        fname="nulls",
    )
    # add Plot to description file
    hh.append_to_html_file(plot_file, desc_html_file)

    # Get correlation


if __name__ == "__main__":
    sys.exit(main())
