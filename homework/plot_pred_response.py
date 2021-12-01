import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from plotly import express as px
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
        plot_paths = []
        for pred in predictors:
            if self.feature_type_dict.get(response) == "continuous":
                if self.feature_type_dict.get(pred) == "continuous":
                    plot_paths.append(self.cont_resp_cont_pred(response, pred))
                else:
                    plot_paths.append(self.cont_resp_cat_pred(response, pred))
            else:
                if self.feature_type_dict.get(pred) == "continuous":
                    plot_paths.append(self.cat_resp_cont_pred(response, pred))
                else:
                    plot_paths.append(self.cat_resp_cat_pred(response, pred))
        return plot_paths

    def cat_resp_cont_pred(self, response, predictor):
        df_plot_temp = self.df[self.df[predictor].notnull()]
        categories = df_plot_temp[response].unique()
        x1 = df_plot_temp[predictor][df_plot_temp[response] == categories[0]]
        x2 = df_plot_temp[predictor][df_plot_temp[response] == categories[1]]
        categories = df_plot_temp[response].unique()

        hist_data = [x1, x2]

        group_labels = [
            "Response = " + str(categories[0]),
            "Response = " + str(categories[1]),
        ]

        fig = go.Figure()
        for curr_hist, curr_group in zip(hist_data, group_labels):
            fig.add_trace(
                go.Violin(
                    x=np.repeat(curr_group, len(curr_group)),
                    y=curr_hist,
                    name=curr_group,
                    box_visible=True,
                    meanline_visible=True,
                )
            )
        fig.update_layout(
            title=response + " by " + predictor,
            xaxis_title=response,
            yaxis_title=predictor,
        )
        file_path = "../output/" + response + "by" + predictor + ".html"
        fig.write_html(
            file=file_path,
            include_plotlyjs="cdn",
        )
        return file_path

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
        file_path = "../output/" + response + "by" + predictor + ".html"
        distribution_plot.write_html(
            file=file_path,
            include_plotlyjs="cdn",
        )
        return file_path

    def cat_resp_cat_pred(self, response, predictor):
        dum_resp_df = pd.get_dummies(self.df[response])
        dum_pred_df = pd.get_dummies(self.df[predictor])

        conf_matrix = confusion_matrix(dum_resp_df.iloc[:, 0], dum_pred_df.iloc[:, 0])

        fig = go.Figure(data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max()))

        fig.update_layout(
            title=response + " by " + predictor,
            xaxis_title=response,
            yaxis_title=predictor,
        )
        file_path = "../output/" + response + "by" + predictor + ".html"
        fig.write_html(
            file=file_path,
            include_plotlyjs="cdn",
        )
        return file_path

    def cont_resp_cont_pred(self, response, predictor):
        x = self.df[predictor]
        y = self.df[response]

        fig = px.scatter(x=x, y=y, trendline="ols")

        fig.update_layout(
            title=response + " by " + predictor,
            xaxis_title=predictor,
            yaxis_title=response,
        )
        file_path = "../output/" + response + "by" + predictor + ".html"
        fig.write_html(
            file=file_path,
            include_plotlyjs="cdn",
        )
        return file_path

    @staticmethod
    def plot_diff_with_mor(df_bins, response, pred):

        bin_cat_list = df_bins["Bin"].tolist()
        cat = []
        for num in bin_cat_list:
            cat.append(str(round(num.mid, 3)))

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
            secondary_y=True,
        )

        ave = df_bins["PopMean"].mean()

        # fig.add_hline(y=2, line_dash='dash', line_color='Red', yref='paper')
        fig.add_shape(
            type="line",
            yref="y2",
            x0=-0.5,
            y0=ave,
            x1=len(bin_cat_list) - 0.5,
            y1=ave,
            line=dict(color="cyan"),
            name="Average",
        )

        fig.update_layout(
            title_text="Binned Difference with Mean of Response vs Bin <br><sup>Response,Predictor: "
            + response
            + ","
            + pred
        )

        fig.show()
        fig.write_html(
            file="../output/" + response + "by" + pred + ".html",
            include_plotlyjs="cdn",
        )

    @staticmethod
    def plot_simple_bar(x, y, x_title="", y_title="", title="", fname=""):
        x = x
        y = y

        fig = go.Figure(
            data=[
                go.Bar(
                    x=x,
                    y=y,
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
        )
        file_path = "../output/" + fname + ".html"
        fig.write_html(file=file_path, include_plotlyjs="cdn")
        return file_path
