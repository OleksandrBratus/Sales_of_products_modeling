# -*- coding: utf-8 -*-

from rest.controllers.controllers import *
from rest.controllers.controllers import __index
from flask import current_app as app


@app.route('/')
def __index_():
    return __index()


@app.route('/run')
def run():
    return run_all()


@app.route('/get_data')
def get_data():
    return get_data_from_db()


@app.route('/preprocess_data')
def load_data():
    return preprocess_data()


@app.route('/train')
def train():
    return train_model()


@app.route('/predict/<sku>/'+u'<area>'+'/'+u'<distr>')
def get_predict(sku, area, distr):
    return get_result(str(sku), str(area), str(distr))
