# -*- coding: utf-8 -*-

import pandas as pd
from model.utils import result
from main import data_from_db, data_preprocessing, train_predict


def __index():
    """
    Start page
    """
    return 'The service is running!'


def get_result(sku, area, distr, path='predictions.csv'):
    """
    Get forecast for the next 12 months from INITIAL_DATE.
    :param sku: product name
    :param area: region in Ukraine
    :param distr: name of distributor
    :return: list with 12 numbers
    """
    df = pd.read_csv(path)
    return result(df, sku, area, distr, 2)


def get_data_from_db():
    """
    Load data from databases.
    """
    data_from_db()
    return 'Data is updated from databases!'


def preprocess_data():
    """
    Data preprocessing for further entry into the models.
    """
    dt = data_preprocessing()
    return 'Data is preprocessed!'


def train_model(path='prepr_data.csv', out='predictions.csv'):
    """
    Models training, postprocessing.
    """
    df = pd.read_csv(path)
    dt = train_predict(df, output=out)
    return 'Models are trained!'


def run_all():
    """
    Data updating, preprocessing and models training.
    """
    data_from_db()
    orders = data_preprocessing()
    data = train_predict(orders)
    return 'Ready for getting forecasts!'
