import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def is_response_boolean(df, resp):
    response = resp
    if str(df[response].dtype) == "object":
        label_encoder = LabelEncoder()
        df[response + "_encoded"] = label_encoder.fit_transform(df[response])
        response = response + "_encoded"
        return response
    else:
        return response


def get_bin_attributes(df, response):

    df_mean_sq_diff = df[["Bin", response]].groupby("Bin").mean()
    df_mean_sq_diff.reset_index(inplace=True)
    df_mean_sq_diff = df_mean_sq_diff.rename(columns={response: "RespBinMean"})
    df_mean_sq_diff["RespPopMean"] = df[response].mean()
    df_bin_counts = (
        df["Bin"].value_counts().rename_axis("Bin").reset_index(name="BinPop")
    )
    df_mean_sq_diff = df_mean_sq_diff.merge(df_bin_counts, on="Bin")
    df_mean_sq_diff["MeanSquaredDiff"] = (
        df_mean_sq_diff["RespBinMean"] - df_mean_sq_diff["RespPopMean"]
    ) ** 2
    df_mean_sq_diff["WeighMeanSquaredDiff"] = (
        ((df_mean_sq_diff["RespBinMean"] - df_mean_sq_diff["RespPopMean"]) ** 2)
        * df_mean_sq_diff["BinPop"]
        / df_bin_counts["BinPop"].sum()
    )

    df_mean_sq_diff = df_mean_sq_diff.sort_values(
        by=["WeighMeanSquaredDiff"], ascending=False
    ).reset_index(drop=True)

    return df_mean_sq_diff


class BinResponseByPredictor:
    def __init__(self, dataframe, feature_type_dictionary, bin_count=None):
        self.df = dataframe.copy()
        self.feature_type_dict = feature_type_dictionary
        self.bin_count = bin_count

    def bin_response_by_predictors(self, response, predictors):
        for pred in predictors:
            if self.feature_type_dict.get(response) == "continuous":
                if self.feature_type_dict.get(pred) == "continuous":
                    return self.bin_cont_resp_cont_pred(response, pred, self.bin_count)
                else:
                    return self.bin_cont_resp_cat_pred(response, pred, self.bin_count)
            else:
                if self.feature_type_dict.get(pred) == "boolean":
                    return self.bin_cat_resp_cat_pred(response, pred, self.bin_count)
                else:
                    return self.bin_cat_resp_cont_pred(response, pred, self.bin_count)

    def bin_2d_response_by_predictors(self, resp, pred1, pred2, bin_counts):
        if (
            self.feature_type_dict.get(pred1) == "continuous"
            and self.feature_type_dict.get(pred1) == "continuous"
        ):
            return self.bin_2d_cont_cont_pred(resp, pred1, pred2, bin_counts)
        elif (
            self.feature_type_dict.get(pred1) == "categorical"
            and self.feature_type_dict.get(pred1) == "categorical"
        ):
            return self.bin_2d_cat_cat_pred(resp, pred1, pred2)
        else:
            return self.bin_2d_cat_cont_pred(resp, pred1, pred2, bin_counts)

    def bin_cont_resp_cont_pred(self, response, pred, bin_counts=None):

        if bin_counts is None:
            bin_counts = self.bin_count
        # Cut df by 10 bins using predictor
        self.df["bin_cat"], bin_array = pd.cut(
            x=self.df[pred], bins=bin_counts, retbins=True
        )

        bin_center = []
        for indx, val in enumerate(bin_array):
            if indx != 0:
                bin_center.append(round((val + bin_array[indx - 1]) / 2, 3))
        predictor = "bin_cat"
        # Find bin categories
        bins_cat = self.df["bin_cat"].sort_values().unique()
        # Get pop mean of response
        pop_mean = self.df[response].mean()
        # create data frame from temp list
        bin_df = self._get_bin_resp_pred_df(
            bins_cat, pop_mean, response, predictor, bin_center
        )
        # drop temporary bin category column
        self.df = self.df.drop(columns=["bin_cat"], axis=1)

        return bin_df

    def bin_cont_resp_cat_pred(self, response, pred, bin_counts=None):

        if bin_counts is None:
            bin_counts = self.bin_count
        # Find bin categories
        bins_cat = self.df[pred].unique()
        # Get pop mean of response
        pop_mean = self.df[response].mean()
        # create data frame from temp list
        bin_df = self._get_bin_resp_pred_df(bins_cat, pop_mean, response, pred)

        return bin_df

    def bin_cat_resp_cat_pred(self, response, pred, bin_counts=None):

        if bin_counts is None:
            bin_counts = self.bin_count
        # dummy response
        dum_resp_df = pd.get_dummies(self.df[response])
        self.df["DummyResponse"] = dum_resp_df.iloc[:, 0]
        # Find bin categories
        bins_cat = self.df[pred].unique()
        # Get pop mean of response
        pop_mean = self.df["DummyResponse"].mean()
        # create data frame from temp list
        bin_df = self._get_bin_resp_pred_df(bins_cat, pop_mean, "DummyResponse", pred)
        # Drop temporary dummy response column
        self.df = self.df.drop(columns=["DummyResponse"], axis=1)

        return bin_df

    def bin_cat_resp_cont_pred(self, response, pred, bin_counts=None):

        if bin_counts is None:
            bin_counts = self.bin_count
        # dummy response
        dum_resp_df = pd.get_dummies(self.df[response])
        self.df["DummyResponse"] = dum_resp_df.iloc[:, 0]

        # Cut df by 10 bins using predictor
        self.df["bin_cat"], bin_array = pd.cut(x=self.df[pred], bins=10, retbins=True)

        # Find center or each bin
        bin_center = []
        for indx, val in enumerate(bin_array):
            if indx != 0:
                bin_center.append(round((val + bin_array[indx - 1]) / 2, 5))

        # Find bin categories
        bins_cat = self.df["bin_cat"].sort_values().unique()
        # Get pop mean of response
        pop_mean = self.df["DummyResponse"].mean()
        # create data frame from temp list
        bin_df = self._get_bin_resp_pred_df(
            bins_cat, pop_mean, "DummyResponse", "bin_cat", bin_center
        )
        # Drop temporary dummy response column
        self.df = self.df.drop(columns=["DummyResponse", "bin_cat"], axis=1)

        return bin_df

    def bin_2d_cont_cont_pred(self, resp, pred1, pred2, bin_counts=None):

        response = is_response_boolean(self.df, resp)

        self.df[pred1 + "Bin"], bin1 = pd.cut(
            x=self.df[pred1], bins=bin_counts, retbins=True
        )
        self.df[pred2 + "Bin"], bin2 = pd.cut(
            x=self.df[pred2], bins=bin_counts, retbins=True
        )

        self.df["Bin"] = (
            self.df[pred1 + "Bin"].astype(str)
            + ","
            + self.df[pred2 + "Bin"].astype(str)
        )

        df_mean_sq_diff = get_bin_attributes(self.df, response)

        if str(self.df[response].dtype) == "object":
            self.df = self.df.drop(
                columns=[pred1 + "Bin", pred2 + "Bin", "Bin", response + "_encoded"],
                axis=1,
            )
        else:
            self.df = self.df.drop(
                columns=[pred1 + "Bin", pred2 + "Bin", "Bin"], axis=1
            )

        return df_mean_sq_diff, bin1, bin2

    def bin_2d_cat_cont_pred(self, resp, pred_cat, pred_cont, bin_counts=None):
        response = is_response_boolean(self.df, resp)

        self.df[pred_cont + "Bin"], pred2_bin_list = pd.cut(
            x=self.df[pred_cont], bins=bin_counts, retbins=True
        )

        self.df["Bin"] = (
            self.df[pred_cat].astype(str) + "," + self.df[pred_cont + "Bin"].astype(str)
        )

        df_mean_sq_diff = get_bin_attributes(self.df, response)

        pred1_bin_list = np.sort(self.df[pred_cat].astype(str).unique())

        if str(self.df[response].dtype) == "object":
            self.df = self.df.drop(
                columns=[pred_cont + "Bin", "Bin", response + "_encoded"], axis=1
            )
        else:
            self.df = self.df.drop(columns=[pred_cont + "Bin", "Bin"], axis=1)

        return df_mean_sq_diff, pred1_bin_list, pred2_bin_list

    def bin_2d_cat_cat_pred(self, resp, pred1, pred2, bin_counts=None):

        response = is_response_boolean(self.df, resp)

        self.df["Bin"] = self.df[pred1].astype(str) + "," + self.df[pred2].astype(str)

        df_mean_sq_diff = get_bin_attributes(self.df, response)

        pred1_bin_list = np.sort(self.df[pred1].astype(str).unique())
        pred2_bin_list = np.sort(self.df[pred2].astype(str).unique())

        return df_mean_sq_diff, pred1_bin_list, pred2_bin_list

    def _get_bin_resp_pred_df(
        self, bins_cat, pop_mean, response, predictor, bin_center=None
    ):

        if bin_center is None:
            bin_center = ["NA" for i in range(10)]

        data_frame_list = []
        data_frame_columns = [
            "Bin",
            "Center",
            "Counts",
            "Means",
            "PopMean",
            "MeanSqrDiff",
            "PopProportion",
            "MeanSqrDiffWeighted",
        ]
        # create rows of each bin with columns from data_frame_columns
        for cat, binCenter in zip(bins_cat, bin_center):
            temp_list = []
            temp_list.append(cat)
            temp_list.append(binCenter)
            bin_pop = self.df[response][(self.df[predictor] == cat)].count()
            temp_list.append(bin_pop)
            temp_list.append(self.df[response][(self.df[predictor] == cat)].mean())
            temp_list.append(pop_mean)
            msd = (
                (self.df[response][(self.df[predictor] == cat)].mean() - pop_mean) ** 2
            ) / bin_pop
            temp_list.append(msd)
            pop_prop = pop_mean / len(self.df.index)
            temp_list.append(pop_prop)
            temp_list.append(
                pop_prop
                * (self.df[response][(self.df[predictor] == cat)].mean() - pop_mean)
                ** 2
            )
            # add to temp list
            data_frame_list.append(temp_list)
        # create data frame from temp list
        bin_df = pd.DataFrame(data_frame_list, columns=data_frame_columns)

        return bin_df
