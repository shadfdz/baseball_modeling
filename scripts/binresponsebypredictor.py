import pandas as pd


class BinResponseByPredictor:
    def __init__(self, Dataframe):
        self.df = Dataframe

    def bin_cont_resp_cont_pred(self, response, pred):

        # Cut df by 10 bins using predictor
        self.df["bin_cat"], bin_array = pd.cut(x=self.df[pred], bins=10, retbins=True)

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

    def bin_cont_resp_cat_pred(self, response, pred):
        # Find bin categories
        bins_cat = self.df[pred].unique()
        # Get pop mean of response
        pop_mean = self.df[response].mean()
        # create data frame from temp list
        bin_df = self._get_bin_resp_pred_df(bins_cat, pop_mean, response, pred)

        return bin_df

    def bin_cat_resp_cat_pred(self, response, pred):
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

    def bin_cat_resp_cont_pred(self, response, pred):
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
            bins_cat, pop_mean, "DummyResponse", "bins_cat", bin_center
        )
        # Drop temporary dummy response column
        self.df = self.df.drop(columns=["DummyResponse", "bin_cat"], axis=1)

        return bin_df

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
            ) / 2
            temp_list.append(msd)
            pop_prop = pop_mean / len(self.df.index)
            temp_list.append(pop_prop)
            temp_list.append(pop_prop * msd)
            # add to temp list
            data_frame_list.append(temp_list)
        # create data frame from temp list
        print("here")
        bin_df = pd.DataFrame(data_frame_list, columns=data_frame_columns)

        return bin_df
