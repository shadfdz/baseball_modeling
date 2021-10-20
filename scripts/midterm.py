import sys

import pandas as pd


def get_response_predictors(dataframe):
    print("The following are features of the dataset: ")
    print(*dataframe.columns)
    response_feature = str(
        input("\nPlease enter the column name of the response variable: ")
    )
    resp = response_feature
    preds = dataframe.loc[:, ~dataframe.columns.isin(resp)].columns.to_list()
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
        if dataframe[col].dtype == "int64":
            if dataframe[col].unique().size == 2:
                feature_type_dict[col] = "categorical"
            else:
                feature_type_dict[col] = "continuous"
        elif dataframe[col].dtype == "float64":
            feature_type_dict[col] = "continuous"
        else:
            feature_type_dict[col] = "categorical"
    return feature_type_dict


def main():
    # get data
    df = pd.read_csv("scripts/titanic.csv")

    # get list of response and predictors
    response, predictors = get_response_predictors(df)

    # split predictors to categorical and continuous and add to respective list
    predictor_type_dict = get_predictor_type_dict(df)
    cat_predictors = []
    cont_predictors = []
    for key in predictor_type_dict.keys():
        if key == "continuous":
            cont_predictors.append(key)
        else:
            cat_predictors.append(key)

    # get correlation for cont/cont predictors


if __name__ == "__main__":

    sys.exit(main())
