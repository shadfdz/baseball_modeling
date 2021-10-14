# import sys

# import numpy
# import pandas as pd
# from plotly import express as px
from plotly import figure_factory as ff

# from plotly import graph_objects as go
# from sklearn.metrics import confusion_matrix

# one main call function that takes the input and determines which function to use to plot


class PlotPredictorResponse:
    def __init__(self, dataframe, feature_type_dictionary):
        self.df = dataframe
        self.df_feat_type_dict = feature_type_dictionary

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

        # fig_2 = go.Figure()
        # for curr_hist, curr_group in zip(hist_data, categories):
        #     fig_2.add_trace(
        #         go.Violin(
        #             x=numpy.repeat(curr_group, len(hist_data)),
        #             y=curr_hist,
        #             name=curr_group,
        #             box_visible=True,
        #             meanline_visible=True,
        #         )
        #     )
        # fig_2.update_layout(
        #     title="Continuous Response by Categorical Predictor",
        #     xaxis_title="Groupings",
        #     yaxis_title="Response",
        # )
        # fig_2.show()
        # fig_2.write_html(
        #     file="../../../plots/lecture_6_cont_response_cat_predictor_violin_plot.html",
        #     include_plotlyjs="cdn",
        # )
        return
