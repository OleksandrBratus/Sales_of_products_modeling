import pandas as pd
from data import load_data
from preprocessing.data_preprocessor import DataPreprocessor
from model.lgbm import LightModel
from model.time_series import TimeSeriesModel
from model.utils import correct, result
from data.data_from_db import load_orders, load_products


def data_from_db():
    orders = load_orders()
    products = load_products()


def data_preprocessing(path="prepr_data.csv"):
    df, products = load_data.load_datasets()
    dp = DataPreprocessor()
    df = dp.preprocess(df, products)
    df.to_csv(path, index=False)
    return df


def train_predict(df, output="predictions.csv"):
    lm = LightModel(df)
    lm.train()
    dt = lm.predict(output)
    dt = correct(dt, df, output)
    ts = TimeSeriesModel(dt, df, output)
    dt = ts.train_predict()
    return dt
