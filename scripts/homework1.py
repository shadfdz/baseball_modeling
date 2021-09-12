# HW 1
# Shad Fernandez
# BDA 696 ML Engineering
# 11-SEP-2021

# import necessary modules
import pandas as pd

# read iris dataframe
iris_df = pd.read_csv(
    "../datasets/iris.data",
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
)
