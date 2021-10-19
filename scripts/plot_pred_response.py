import pandas as pd
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix


class PlotPredictorResponse:
    """
    The class takes a data frame and a dictionary of columns containing each
    data type
    """

    def __init__(self, dataframe, feature_type_dictionary):
        self.df = dataframe
        self.feature_type_dict = feature_type_dictionary

    def plot_response_by_predictors(self, response, predictors):

        for pred in predictors:
            if self.feature_type_dict.get(response) == "continuous":
                if self.feature_type_dict.get(pred) == "continuous":
                    self.cont_resp_cont_pred(response, pred)
                else:
                    self.cont_resp_cat_pred(response, pred)
            else:
                if self.feature_type_dict.get(pred) == "boolean":
                    self.cat_resp_cat_pred(response, pred)
                else:
                    self.cat_resp_cont_pred(response, pred)

    def cat_resp_cont_pred(self, response, predictor):

        categories = self.df[response].unique()

        for cat in categories:
            x1 = self.df[predictor][(self.df[response] == cat)]
            x2 = self.df[predictor][(self.df[response] != cat)]

            hist_data = [x1, x2]
            group_labels = ["Response = not " + cat, "Response = " + cat]

            fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
            fig_1.update_layout(
                title=predictor + " by " + cat + " as response",
                xaxis_title=predictor,
                yaxis_title="Distribution",
            )
            fig_1.show()
            fig_1.write_html(
                file="../plots/" + cat + "by" + predictor + ".html",
                include_plotlyjs="cdn",
            )

    def cont_resp_cat_pred(self, response, predictor):

        # get categories and store in a list
        categories = self.df[predictor].unique()

        # store response values of each categories in a list
        hist_data = []
        for cat in categories:
            hist_data.append(self.df[response][(self.df[predictor] == cat)])

        # Create distribution plot with custom bin_size
        distribution_plot = ff.create_distplot(hist_data, categories, bin_size=0.2)
        distribution_plot.update_layout(
            title=response + " by " + predictor,
            xaxis_title=response,
            yaxis_title="Distribution",
        )
        distribution_plot.show()
        distribution_plot.write_html(
            file="../plots/" + response + "by" + predictor + ".html",
            include_plotlyjs="cdn",
        )

    def cat_resp_cat_pred(self, response, predictors):
        dum_resp_df = pd.get_dummies(self.df[response])
        dum_pred_df = pd.get_dummies(self.df[predictors])

        conf_matrix = confusion_matrix(dum_resp_df.iloc[:, 0], dum_pred_df.iloc[:, 0])

        fig_no_relationship = go.Figure(
            data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
        )

        fig_no_relationship.update_layout(
            title=response + predictors + " (without relationship)",
            xaxis_title=response,
            yaxis_title=predictors,
        )
        fig_no_relationship.show()
        fig_no_relationship.write_html(
            file="../plots/" + response + "by" + predictors + ".html",
            include_plotlyjs="cdn",
        )

    def cont_resp_cont_pred(self, response, predictor):
        x = self.df[predictor]
        y = self.df[response]

        fig = px.scatter(x=x, y=y, trendline="ols")

        fig.update_layout(
            title=response + " by " + predictor,
            xaxis_title=predictor,
            yaxis_title=response,
        )
        fig.show()
        fig.write_html(
            file="../plots/" + response + "by" + predictor + ".html",
            include_plotlyjs="cdn",
        )

    def plot_diff_with_MOR(self, df_bins, response, pred):

        bin_cat_list = df_bins["Bin"].tolist()
        cat = []
        for num in bin_cat_list:
            cat.append(str(num))

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=cat, y=df_bins["Counts"], name="Population"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=cat,
                y=df_bins["Means"] - df_bins["PopMean"],
                name="BinMeanResponse-PopMeanResponse",
            ),
            secondary_y="True",
        )

        fig.update_layout(
            title_text="Binned Difference with Mean of Response vs Bin <br><sup>Response,Predictor: "
            + response
            + ","
            + pred
        )

        fig.show()
        fig.write_html(
            file="../plots/" + response + "by" + pred + ".html",
            include_plotlyjs="cdn",
        )
