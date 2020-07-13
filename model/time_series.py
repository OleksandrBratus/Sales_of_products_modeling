import pandas as pd
import numpy as np
from datetime import date
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from preprocessing.data_preprocessor import interpol
from model.utils import add_predict
from config.config import INITIAL_DATE


class TimeSeriesModel:

    def __init__(self, data, extended_dataset, output):
        self.data = data
        self.extended_dataset = extended_dataset
        self.output = output

    def train_predict(self):
        data = add_predict(active_data=self.data,
                           orders=self.extended_dataset,
                           func=self._ts_predict,
                           period=INITIAL_DATE,
                           smokie=True,
                           name="_apricot&pineapple asorti",
                           output=self.output,
                           threshold=100)
        data = add_predict(active_data=data,
                           orders=self.extended_dataset,
                           func=self._ts_predict,
                           period=INITIAL_DATE,
                           smokie=True,
                           name="_pineapple&carrots&nuts",
                           output=self.output,
                           threshold=100)
        data = add_predict(active_data=data,
                           orders=self.extended_dataset,
                           func=self._ts_predict,
                           period=INITIAL_DATE,
                           smokie=True,
                           name="beertail_",
                           output=self.output,
                           threshold=50)
        data = add_predict(active_data=data,
                           orders=self.extended_dataset,
                           func=self._ts_predict,
                           period=INITIAL_DATE,
                           smokie=True,
                           name="btl_4,8%_wild lemon",
                           output=self.output,
                           threshold=100)
        data = add_predict(active_data=data,
                           orders=self.extended_dataset,
                           func=self._ts_predict,
                           period=INITIAL_DATE,
                           smokie=True,
                           name="can_4,8%_wild lemon",
                           output=self.output,
                           threshold=100)
        data = add_predict(active_data=data,
                           orders=self.extended_dataset,
                           func=self._zak_undef_predict,
                           period=INITIAL_DATE,
                           name="Закарпатская",
                           output=self.output)
        data = add_predict(active_data=data,
                           orders=self.extended_dataset,
                           func=self._zak_undef_predict,
                           period=INITIAL_DATE,
                           name="Undef",
                           output=self.output)
        return data

    @staticmethod
    def _ts_predict(orders, sku, area, distr, period='2019', threshold=100):

        df = interpol(orders, sku, area, distr, extra=True)

        mem = 0
        df["Date_ID"] = df.year.astype("str") + "-" + df.month.astype("str") + "-01"
        df["Date_ID"] = df["Date_ID"].astype(np.datetime64)
        dates = pd.DataFrame(
            {"Date_ID": pd.date_range(start=f"{df.year.min()}-{df[df.year == df.year.min()].month.min()}-01",
                                      end=f"{df.year.max()}-{df[df.year == df.year.max()].month.max()}-01")})
        df = dates.merge(df[["Date_ID", "QTY"]], how="left", on="Date_ID").fillna(0)
        del dates
        df["month"] = df['Date_ID'].values.astype('datetime64[M]')
        dff = pd.DataFrame(df.groupby("month").QTY.sum())
        dff["pred"] = dff.QTY

        flag = False

        if period == '2019':
            current_year = int(period)
            current_month = '01'
            if dff[dff.index.year < int(period)].QTY.values[-3:].mean() < threshold:
                flag = True
                mem = dff[dff.index.year < int(period)].QTY.values[-2:].mean()
        elif period == 'today':
            today = date.today()
            current_year = today.year
            current_month = today.month
            if dff[(dff.index.year < today.year - 1) |
                   ((dff.index.year == today.year - 1) & (dff.index.month < today.month))].QTY.values[
               -3:].mean() < threshold:
                flag = True
                mem = dff[(dff.index.year < today.year - 1) |
                          ((dff.index.year == today.year - 1) & (dff.index.month < today.month))].QTY.values[-2:].mean()
        else:
            current_year = period.year
            current_month = period.month
            if dff[(dff.index.year < current_year) |
                   ((dff.index.year == current_year) & (dff.index.month < current_month))].QTY.values[
               -3:].mean() < threshold:
                flag = True
                mem = dff[(dff.index.year < current_year) |
                          ((dff.index.year == current_year) & (dff.index.month < current_month))].QTY.values[-2:].mean()

        try:
            series = dff.QTY
            size = np.sum(series.index >= f"{current_year}-{current_month}-01")

            if size == 0:
                train, test = series[:], series[:0]
                return None
            else:
                train, test = series[:-size], series[-size:]

            if train.shape[0] >= 18:
                a = '18_2'
                model = SARIMAX(train, order=(3, 1, 3), seasonal_order=(3, 0, 0, 0))
            elif train.shape[0] >= 12:
                a = '12_2'
                model = SARIMAX(train, order=(3, 0, 1), seasonal_order=(3, 0, 0, 0))
            elif train.shape[0] >= 6:
                a = '6_2'
                model = ARIMA(train, order=(3, 1, 2))
            elif train.shape[0] >= 3:
                a = '3_2'
                model = ARIMA(train, order=(2, 0, 0))
            else:
                model = ARIMA(train, order=(1, 0, 0))
            fit_model = model.fit(disp=True)
            pred = fit_model.predict(start=test.index.min(), end=test.index.max(), dynamic=True)
            dff["pred"] = dff.QTY
            dff.loc[pred.index, "pred"] = np.maximum(0, np.round(pred))
        except:
            try:
                series = dff.QTY
                size = np.sum(series.index >= f"{current_year}-{current_month}-01")

                if size == 0:
                    train, test = series[:], series[:0]
                    return None
                else:
                    train, test = series[:-size], series[-size:]

                model = ARIMA(train, order=(2, 0, 0))
                fit_model = model.fit(disp=True)
                pred = fit_model.predict(start=test.index.min(), end=test.index.max(), dynamic=True)
                dff["pred"] = dff.QTY
                dff.loc[pred.index, "pred"] = np.maximum(0, np.round(pred))
                a = 'ARIMA'
            except:
                dff['pred'] = dff[dff.index.year < 2019]["QTY"].values[-3:].mean()
                a = 'Oops...'

        if flag:
            dff['pred'] = mem

        res = dff.reset_index()
        res["month_start"] = res['month'].values.astype('datetime64[M]')
        res = res.groupby("month_start")["QTY", "pred"].sum().reset_index()
        res["SKU"] = sku
        res["Area"] = area
        res["DISTR_NAME"] = distr
        res['month'] = res.month_start.dt.month
        res['year'] = res.month_start.dt.year

        return res

    @staticmethod
    def _zak_undef_predict(orders, sku, area, distr, period='2019', threshold=100):

        df = interpol(orders, sku, area, distr, extra=True)

        mem = 0
        df["Date_ID"] = df.year.astype("str") + "-" + df.month.astype("str") + "-01"
        df["Date_ID"] = df["Date_ID"].astype(np.datetime64)
        dates = pd.DataFrame(
            {"Date_ID": pd.date_range(start=f"{df.year.min()}-{df[df.year == df.year.min()].month.min()}-01",
                                      end=f"{df.year.max()}-{df[df.year == df.year.max()].month.max()}-01")})
        df = dates.merge(df[["Date_ID", "QTY"]], how="left", on="Date_ID").fillna(0)
        del dates
        df["month"] = df['Date_ID'].values.astype('datetime64[M]')
        dff = pd.DataFrame(df.groupby("month").QTY.sum())
        dff["pred"] = dff.QTY

        flag = False

        if period == '2019':
            current_year = int(period)
            current_month = '01'
            if dff[dff.index.year < int(period)].QTY.values[-3:].mean() < threshold:
                flag = True
                mem = dff[dff.index.year < int(period)].QTY.values[-2:].mean()
        elif period == 'today':
            today = date.today()
            current_year = today.year
            current_month = today.month
            if dff[(dff.index.year < today.year - 1) |
                   ((dff.index.year == today.year - 1) & (dff.index.month < today.month))].QTY.values[
               -3:].mean() < threshold:
                flag = True
                mem = dff[(dff.index.year < today.year - 1) |
                          ((dff.index.year == today.year - 1) & (dff.index.month < today.month))].QTY.values[-2:].mean()
        else:
            current_year = period.year
            current_month = period.month
            if dff[(dff.index.year < current_year) |
                   ((dff.index.year == current_year) & (dff.index.month < current_month))].QTY.values[
               -3:].mean() < threshold:
                flag = True
                mem = dff[(dff.index.year < current_year) |
                          ((dff.index.year == current_year) & (dff.index.month < current_month))].QTY.values[-2:].mean()

        try:
            series = dff.QTY
            size = np.sum(series.index >= f"{current_year}-{current_month}-01")

            if size == 0:
                train, test = series[:], series[:0]
                return None
            else:
                train, test = series[:-size], series[-size:]

            if train.shape[0] >= 18:
                a = '18_2'
                model = SARIMAX(train, order=(3, 1, 3), seasonal_order=(3, 0, 0, 0))
            elif train.shape[0] >= 12:
                a = '12_2'
                model = SARIMAX(train, order=(3, 0, 1), seasonal_order=(3, 0, 0, 0))
            elif train.shape[0] >= 6:
                a = '6_2'
                model = ARIMA(train, order=(3, 1, 2))
            elif train.shape[0] >= 3:
                a = '3_2'
                model = ARIMA(train, order=(2, 0, 0))
            else:
                model = ARIMA(train, order=(1, 0, 0))
            fit_model = model.fit(disp=True)
            pred = fit_model.predict(start=test.index.min(), end=test.index.max(), dynamic=True)
            dff["pred"] = dff.QTY
            dff.loc[pred.index, "pred"] = np.maximum(0, np.round(pred))
        except:
            try:
                series = dff.QTY
                size = np.sum(series.index >= f"{current_year}-{current_month}-01")

                if size == 0:
                    train, test = series[:], series[:0]
                    return None
                else:
                    train, test = series[:-size], series[-size:]
                model = ARIMA(train, order=(2, 0, 0))
                fit_model = model.fit(disp=True)
                pred = fit_model.predict(start=test.index.min(), end=test.index.max(), dynamic=True)
                dff["pred"] = dff.QTY
                dff.loc[pred.index, "pred"] = np.maximum(0, np.round(pred))
                a = "ARIMA"
            except:
                a = 'Oops...'
                dff['pred'] = dff[dff.index.year < 2019]["QTY"].values[-3:].mean()

        if flag:
            dff['pred'] = mem

        res = dff.reset_index()
        res["month_start"] = res['month'].values.astype('datetime64[M]')
        res = res.groupby("month_start")["QTY", "pred"].sum().reset_index()
        res["SKU"] = sku
        res["Area"] = area
        res["DISTR_NAME"] = distr
        res['month'] = res.month_start.dt.month
        res['year'] = res.month_start.dt.year

        return res
