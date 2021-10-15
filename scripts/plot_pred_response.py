import pandas as pd
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix

# one main call function that takes the input and determines which function to use to plot


class PlotPredictorResponse:
    def __init__(self, dataframe, feature_type_dictionary):
        self.df = dataframe
        self.df_feat_type_dict = feature_type_dictionary

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
