import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import copy
from datetime import date
from config.config import MAX_DATE, INITIAL_DATE


class DataPreprocessor:
    def preprocess(self, df, products, period=INITIAL_DATE) -> pd.DataFrame:

        df = self._delete_old_distr(df, period=period)
        df = self._delete_one_months(self, df, period=period)
        df = self._delete_without_history(df, period=period)
        df = self._fill_missing_values_predict(self, df, period=period)
        df = self._group_by_kegs(df, products)
        df = self._delete_without_history(df, period=period)
        df = self._data_smoothing(self, df)
        df.loc[(((df.year == MAX_DATE.year) & (df.month > MAX_DATE.month)) |
                (df.year > MAX_DATE.year)), 'QTY'] = 0

        return df

    @staticmethod
    def _delete_old_distr(orders, period, show=False) -> pd.DataFrame:
        all_dist = np.unique(orders.DISTR_NAME)
        if period == '2019':
            act_dist = np.unique(orders[orders.year == 2019].DISTR_NAME)
        elif period == 'today':
            today = date.today()
            act_dist = np.unique(orders[(orders.year > today.year - 1) |
                                        ((orders.year == today.year - 1) & (orders.month >= today.month))].DISTR_NAME)
        else:
            current_year = period.year
            current_month = period.month
            act_dist = np.unique(orders[(orders.year > current_year) |
                                        ((orders.year == current_year) & (orders.month >= current_month))].DISTR_NAME)
        old_dist = list(set(all_dist) - set(act_dist))
        if show:
            print('Old distributors:\n', old_dist)
        return orders[~orders.DISTR_NAME.isin(old_dist)]

    @staticmethod
    def _delete_one_month(orders, sku, area, distr, period, show=False):

        check = orders[(orders.SKU == sku)][(orders.Area == area)][(orders.DISTR_NAME == distr)]
        df = orders[(orders.SKU == sku)][(orders.Area == area)][(orders.DISTR_NAME == distr)]

        df["Date_ID"] = df.year.astype("str") + "-" + df.month.astype("str") + "-01"
        df["Date_ID"] = df["Date_ID"].astype(np.datetime64)

        if period == '2019':
            if df[df.year < int(period)].shape[0] <= 1:
                if show:
                    print(df)
                return None
        elif period == 'today':
            today = date.today()
            if df[(df.year < today.year - 1) | ((df.year == today.year - 1) & (df.month < today.month))].shape[0] <= 1:
                if show:
                    print(df)
                return None
        else:
            current_year = period.year
            current_month = period.month
            if df[(df.year < current_year) | ((df.year == current_year) & (df.month < current_month))].shape[0] <= 1:
                if show:
                    print(df)
                return None

        df["month"] = df['Date_ID'].values.astype('datetime64[M]')

        dff = pd.DataFrame(df.groupby("month").QTY.sum())

        res = dff.reset_index()
        res["month_start"] = res['month'].values.astype('datetime64[M]')
        res = res.groupby("month_start")["QTY"].sum().reset_index()
        res['year'] = res.month_start.dt.year
        res['month'] = res.month_start.dt.month
        res["SKU"] = sku
        res["Area"] = area
        res["DISTR_NAME"] = distr
        res = res[['year', 'month', 'QTY', 'SKU', 'Area', 'DISTR_NAME']]

        if check.shape != res.shape:
            print(check.shape)
            print(res.shape)
            print('Alarm', sku, area, distr)

        return res

    @staticmethod
    def _delete_one_months(self, orders, period, show=False) -> pd.DataFrame:

        datas = []

        for sku, area, distr in orders[["SKU", "Area", "DISTR_NAME"]].drop_duplicates().values:
            try:
                datas.append(self._delete_one_month(orders, sku, area, distr, period=period))
            except:
                print(sku, area, distr)
                print("Data error!")
                pass

        datass = pd.concat(datas, axis=0).reset_index().drop(columns=['index'])
        if show:
            print('Number of observations before deleting:', orders.shape)
        one_month = pd.concat([orders, datass]).drop_duplicates(keep=False)
        if show:
            print('Number of deleted observations:', one_month.shape)
        orders = datass.copy()
        if show:
            print('Number of observations after deleting:', orders.shape)

        return orders

    @staticmethod
    def _delete_without_history(orders, period, show=False) -> pd.DataFrame:

        not_zeros = []
        zeros = []

        for sku, area, distr in orders[["SKU", "Area", "DISTR_NAME"]].drop_duplicates().values:
            try:
                if period == '2019':
                    if orders[orders.SKU == sku][orders.Area == area][orders.DISTR_NAME == distr][
                        orders.year < 2019].count().sum() != 0:
                        not_zeros.append([sku, area, distr])
                    else:
                        zeros.append([sku, area, distr])
                elif period == 'year':
                    today = date.today()
                    today_year = today.year
                    today_month = today.month
                    if orders[orders.SKU == sku][orders.Area == area][orders.DISTR_NAME == distr][
                        (orders.year < today.year - 1) |
                        ((orders.year == today.year - 1) &
                         (orders.month < today.month))].count().sum() != 0:
                        not_zeros.append([sku, area, distr])
                    else:
                        zeros.append([sku, area, distr])
                else:
                    current_year = period.year
                    current_month = period.month
                    if orders[orders.SKU == sku][orders.Area == area][orders.DISTR_NAME == distr][
                        (orders.year < current_year) |
                        ((orders.year == current_year) &
                         (orders.month < current_month))].count().sum() != 0:
                        not_zeros.append([sku, area, distr])
                    else:
                        zeros.append([sku, area, distr])
            except:
                print(sku, area, distr)

        not_z = pd.DataFrame(not_zeros)
        not_z.columns = ["SKU", "Area", "DISTR_NAME"]
        if show:
            print('Number of observations before deleting sku_area_distr without history:', orders.shape)
        orders = not_z.merge(orders, how='inner', on=["SKU", "Area", "DISTR_NAME"])
        if show:
            print('Number of observations after deleting sku_area_distr without history:', orders.shape)

        return orders

    @staticmethod
    def _fill_missing_value(orders, sku, area, distr, period) -> pd.DataFrame:

        data = []
        df = orders[(orders.SKU == sku) & (orders.Area == area) & (orders.DISTR_NAME == distr) &
                    (((orders.year == period.year + 1) & (orders.month < period.month)) |
                     ((orders.year == period.year) & (orders.month >= period.month)))]
        data_m = orders[(orders.SKU == sku) & (orders.Area == area) & (orders.DISTR_NAME == distr) &
                        ((orders.year < period.year) |
                         ((orders.year == period.year) & (orders.month < period.month)))]

        if MAX_DATE.year == period.year:
            count_expected_months = MAX_DATE.month - period.month + 1
        elif MAX_DATE.year - 1 == period.year:
            count_expected_months = min(12, MAX_DATE.month + 12 - period.month + 1)
        else:
            count_expected_months = 12

        #     print(df.shape[0], count_expected_months)

        if df.shape[0] != count_expected_months:
            mean_QTY = data_m.QTY.mean()
            if np.isnan(mean_QTY):
                print(df.shape[0], count_expected_months)
                print('mean: ', mean_QTY, sku, area, distr)
            example = orders[(orders.SKU == sku) & (orders.Area == area) & (orders.DISTR_NAME == distr)].head(1)
            if period.year < MAX_DATE.year:
                for i in range(period.month, 12 + 1):
                    if df[(df.year == period.year) & (df.month == i)].shape[0] == 0:
                        example.year = period.year
                        example.month = i
                        example.QTY = mean_QTY
                        data.append(copy.deepcopy(example))
                    else:
                        data.append(df[(df.year == period.year) & (df.month == i)])
                if period.year + 1 < MAX_DATE.year:
                    for i in range(1, period.month):
                        if df[(df.year == period.year + 1) & (df.month == i)].shape[0] == 0:
                            example.year = period.year + 1
                            example.month = i
                            example.QTY = mean_QTY
                            data.append(copy.deepcopy(example))
                        else:
                            data.append(df[(df.year == period.year + 1) & (df.month == i)])
                else:
                    for i in range(1, min(period.month, MAX_DATE.month + 1)):
                        if df[(df.year == period.year + 1) & (df.month == i)].shape[0] == 0:
                            example.year = period.year + 1
                            example.month = i
                            example.QTY = mean_QTY
                            data.append(copy.deepcopy(example))
                        else:
                            data.append(df[(df.year == period.year + 1) & (df.month == i)])
            elif period.year == MAX_DATE.year:
                for i in range(period.month, MAX_DATE.month + 1):
                    if df[(df.year == period.year) & (df.month == i)].shape[0] == 0:
                        example.year = period.year
                        example.month = i
                        example.QTY = mean_QTY
                        data.append(copy.deepcopy(example))
                    else:
                        data.append(df[(df.year == period.year) & (df.month == i)])

            if len(data) != count_expected_months:
                print('Error: ', sku, area, distr)
            return pd.concat(data, axis=0)

    @staticmethod
    def _fill_future_value(orders, sku, area, distr, period) -> pd.DataFrame:

        data = []
        example = orders[(orders.SKU == sku) & (orders.Area == area) & (orders.DISTR_NAME == distr)].head(1)

        if ((period.year + 1 < MAX_DATE.year) | (
                (period.year + 1 == MAX_DATE.year) & (period.month + 1 <= MAX_DATE.month))):
            return orders[(orders.SKU == sku) & (orders.Area == area) & (orders.DISTR_NAME == distr) &
                          (((orders.year == period.year) & (orders.month >= period.month)) |
                           ((orders.year == period.year + 1) & (orders.month < period.month)))]
        elif period.year + 1 == MAX_DATE.year:
            for i in range(MAX_DATE.month + 1, period.month):
                example.year = period.year + 1
                example.month = i
                example.QTY = 0
                data.append(copy.deepcopy(example))
        else:
            for i in range(MAX_DATE.month + 1, 12 + 1):
                example.year = period.year
                example.month = i
                example.QTY = 0
                data.append(copy.deepcopy(example))
            for i in range(1, period.month):
                example.year = period.year + 1
                example.month = i
                example.QTY = 0
                data.append(copy.deepcopy(example))
        return pd.concat(data, axis=0)

    @staticmethod
    def _fill_missing_values_predict(self, orders, period, show=False) -> pd.DataFrame:

        values = []
        for sku, area, distr in orders[["SKU", "Area", "DISTR_NAME"]].drop_duplicates().values:
            try:
                values.append(self._fill_missing_value(orders, sku, area, distr, period=period))
            except:
                print('Error! Missing: ', sku, area, distr)
                pass
            try:
                values.append(self._fill_future_value(orders, sku, area, distr, period=period))
            except:
                print('Error! Future: ', sku, area, distr)
                pass
        predict = pd.concat(values, axis=0)
        if show:
            print(predict.shape)
        predict = pd.concat([predict, orders[(orders.year < period.year) |
                                             ((orders.year == period.year) & (orders.month < period.month))]], axis=0)
        return predict

    @staticmethod
    def _smooth(series):
        for i in range(1, series.shape[0] - 1):
            if series[i] == 0:
                if series[i - 1] >= 10 and series[i + 1] >= 10:
                    series[i] = (series[i - 1] + series[i + 1]) / 2
                else:
                    for j in range(1, min(9, series.shape[0] - 1 - i)):
                        if series[i - 1] >= 10 and series[i + j] >= 10:
                            series[i] = (series[i - 1] + series[i + j].max()) / 2
                            break
            else:
                if series[i - 1] / series[i] >= 10 and series[i + 1] / series[i] >= 10:
                    series[i] = (series[i - 1] + series[i + 1]) / 2
                else:
                    for j in range(1, min(9, series.shape[0] - 1 - i)):
                        if series[i - 1] / series[i] >= 10 and series[i + j] / series[i] >= 10:
                            series[i] = (series[i - 1] + series[i + j]) / 2
                            break
        return series

    @staticmethod
    def _data_smoothing(self, orders) -> pd.DataFrame:
        data = []
        for sku, area, distr in orders[["SKU", "Area", "DISTR_NAME"]].drop_duplicates().values:
            slice_ = orders[(orders.SKU == sku) & (orders.Area == area) & (orders.DISTR_NAME == distr)]
            slice_.QTY = self._smooth(slice_.QTY.values)
            data.append(slice_)
        return pd.concat(data)

    @staticmethod
    def _group_by_kegs(detail_df, products) -> pd.DataFrame:

        month_aggs_distr = pd.DataFrame(
            detail_df.groupby(["SKU", "Area", "DISTR_NAME", "year", "month"])["QTY"].mean()).reset_index()
        last_month_aggs_distr = month_aggs_distr[month_aggs_distr.month == 12]
        month_aggs_distr = month_aggs_distr.merge(last_month_aggs_distr[["SKU", "Area", "DISTR_NAME", "year", "QTY"]],
                                                  on=["SKU", "Area", "DISTR_NAME", "year"], suffixes=("", "_end_year"))

        month_aggs_area = pd.DataFrame(detail_df.groupby(["SKU", "Area", "year", "month"])["QTY"].mean()).reset_index()
        month_aggs_sku = pd.DataFrame(detail_df.groupby(["SKU", "year", "month"])["QTY"].mean()).reset_index()

        month_aggs_last_year = pd.DataFrame(
            detail_df.groupby(["SKU", "Area", "DISTR_NAME", "year"])["QTY"].mean()).reset_index()
        month_aggs_last_year.QTY = month_aggs_last_year.QTY // 12

        month_aggs_distr["year"] = month_aggs_distr["year"] + 1
        month_aggs_area["year"] = month_aggs_area["year"] + 1
        month_aggs_sku["year"] = month_aggs_sku["year"] + 1
        month_aggs_last_year["year"] = month_aggs_last_year["year"] + 1

        orders = detail_df.merge(products.drop(["SKU"], axis=1).drop_duplicates(),
                                 how="inner", left_on="SKU", right_on="SKU_NEW").drop(["SKU_NEW"], axis=1)

        orders = orders.merge(month_aggs_sku,
                              on=["SKU", "year", "month"],
                              how="left",
                              suffixes=("", "_month_sku")).fillna(0.0)

        orders = orders.merge(month_aggs_area,
                              on=["SKU", "year", "month", "Area"],
                              how="left",
                              suffixes=("", "_month_area")).fillna(0.0)

        orders = orders.merge(month_aggs_distr,
                              on=["SKU", "year", "month", "Area", "DISTR_NAME"],
                              how="left",
                              suffixes=("", "_month_distr")).fillna(0.0)

        orders = orders.merge(month_aggs_last_year,
                              on=["SKU", "year", "Area", "DISTR_NAME"],
                              how="left",
                              suffixes=("", "_year_distr")).fillna(0.0)

        orders.QTY = np.maximum(0, orders.QTY)
        orders["keg_"] = orders.SKU.str.split("_").apply(lambda x: x[-1])

        return orders


def interpol(orders, sku, area, distr, extra=False, show=False):
    df = orders[(orders.SKU == sku) & (orders.Area == area) & (orders.DISTR_NAME == distr)]

    df["Date_ID"] = df.year.astype("str") + "-" + df.month.astype("str") + "-01"
    df["Date_ID"] = df["Date_ID"].astype(np.datetime64)

    df["month"] = df['Date_ID'].values.astype('datetime64[M]')

    dfff = orders[(orders.SKU == sku) & (orders.Area == area) & (orders.DISTR_NAME == distr)]
    dfff["Date_ID"] = dfff.year.astype("str") + "-" + dfff.month.astype("str") + "-01"
    dfff["Date_ID"] = dfff["Date_ID"].astype(np.datetime64)
    if extra:
        dates = pd.DataFrame({"Date_ID": pd.date_range(start=f"{dfff.year.min()}-01-01",
                                                       end=f"{dfff.year.max()}-{dfff[dfff.year == dfff.year.max()].month.max()}-01")})
    else:
        dates = pd.DataFrame(
            {"Date_ID": pd.date_range(start=f"{dfff.year.min()}-{dfff[dfff.year == dfff.year.min()].month.min()}-01",
                                      end=f"{dfff.year.max()}-{dfff[dfff.year == dfff.year.max()].month.max()}-01")})
    dfff = dates.merge(dfff[["Date_ID", "QTY"]], how="left", on="Date_ID").fillna(0)
    del dates
    dfff["month"] = dfff['Date_ID'].values.astype('datetime64[M]')

    dc = pd.DataFrame(dfff.groupby("month").QTY.sum()).reset_index().reset_index()
    all_sp = np.array(dc['index'])

    datafr = dc[['index', 'month']].merge(df, how="right", left_on=["month"], right_on=["Date_ID"])
    x = np.array(datafr['index'])
    y = np.array(datafr['QTY'])

    if extra:
        z = interp1d(x, y, kind='nearest', fill_value="extrapolate")
    else:
        z = interp1d(x, y, kind='nearest')

    dc.QTY = z(all_sp)

    res = dc.drop(columns=['index'])
    res["month_start"] = res['month'].values.astype('datetime64[M]')
    res = res.groupby("month_start")["QTY"].sum().reset_index()
    res['year'] = res.month_start.dt.year
    res['month'] = res.month_start.dt.month
    res["SKU"] = sku
    res["Area"] = area
    res["DISTR_NAME"] = distr
    res = res[['year', 'month', 'QTY', 'SKU', 'Area', 'DISTR_NAME']]

    return res


def interpolation(orders, extra=False, show=False):
    if show:
        print('Number of observations before interpolation:', orders.shape)

    data_interp = []
    for sku, area, distr in orders[["SKU", "Area", "DISTR_NAME"]].drop_duplicates().values:
        try:
            data_interp.append(interpol(orders, sku, area, distr, extra))
        except:
            print(sku, area, distr)
            print("Data error!")
            pass

    datas_interp = pd.concat(data_interp, axis=0)

    if show:
        print('Number of observations after interpolation:', datas_interp.shape)

    return datas_interp