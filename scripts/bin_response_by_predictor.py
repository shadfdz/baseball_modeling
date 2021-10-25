import pandas as pd


class BinResponseByPredictor:
    def __init__(self, dataframe, feature_type_dictionary, bin_count):
        self.df = dataframe
        self.feature_type_dict = feature_type_dictionary
        self.bin_count = bin_count

    def bin_response_by_predictors(self, response, predictors):
        bins_df_dict = {}
        for pred in predictors:
            if self.feature_type_dict.get(response) == "continuous":
                if self.feature_type_dict.get(pred) == "continuous":
                    bins_df_dict[pred] = self.bin_cont_resp_cont_pred(
                        response, pred, self.bin_count
                    )
                else:
                    bins_df_dict[pred] = self.bin_cont_resp_cat_pred(
                        response, pred, self.bin_count
                    )
            else:
                if self.feature_type_dict.get(pred) == "boolean":
                    bins_df_dict[pred] = self.bin_cat_resp_cat_pred(
                        response, pred, self.bin_count
                    )
                else:
                    bins_df_dict[pred] = self.bin_cat_resp_cont_pred(
                        response, pred, self.bin_count
                    )

        return bins_df_dict

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
                bin_center.append(round((val + bin_array[indx - 1]) / 2, 5))
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

    def bin_2d_cont_resp_cont_pred(
        self, df, response, pred1, pred2, bin_counts=None, weighted=None
    ):

        df["CatAge"], bin_array1 = pd.cut(x=df["Age"], bins=10, retbins=True)

        df["CatFare"], bin_array2 = pd.cut(x=df["Fare"], bins=10, retbins=True)

        df["Join"] = df["CatAge"].astype(str) + "," + df["CatFare"].astype(str)

        # df_temp_heatmap = df[['Join','Age']].groupby('Join').count()
        # df_temp_heatmap['AveAge'] = df[['Join','Age']].groupby('Join').mean()

        df_temp_heatmap = df[["Join", "Fare"]].groupby("Join").count()
        df_temp_heatmap["AveFare"] = df[["Join", "Fare"]].groupby("Join").mean()

        df_temp_heatmap["Resp"] = df[["Join", "Survived"]].groupby("Join").mean()
        df_temp_heatmap.reset_index(inplace=True)
        print(df_temp_heatmap)

        list1 = df["CatAge"].unique().sort_values()
        list2 = df["CatFare"].unique().sort_values()

        df_list = []
        for outer_bin in list1:
            temp_list = []
            for inner_bin in list2:
                bin_array = str(outer_bin) + "," + str(inner_bin)
                val = df_temp_heatmap.loc[df_temp_heatmap["Join"] == bin_array, "Resp"]
                if val.size == 0:
                    temp_list.append(0)
                else:
                    temp_list.append(val.iloc[0])
            df_list.append(temp_list)

        temp_df = pd.DataFrame(df_list, columns=list2, index=list1)
        print(temp_df)
        pass

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
            ) / self.bin_count
            temp_list.append(msd)
            pop_prop = pop_mean / len(self.df.index)
            temp_list.append(pop_prop)
            temp_list.append(pop_prop * msd)
            # add to temp list
            data_frame_list.append(temp_list)
        # create data frame from temp list
        bin_df = pd.DataFrame(data_frame_list, columns=data_frame_columns)

        return bin_df
