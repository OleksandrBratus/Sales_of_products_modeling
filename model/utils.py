import pandas as pd
import numpy as np
from datetime import date
from collections import OrderedDict
from preprocessing.data_preprocessor import interpol
from config.config import INITIAL_DATE


def correct(data, extended_dataset, output):
    data = add_predict(active_data=data,
                       orders=extended_dataset,
                       func=apply_threshold,
                       period=INITIAL_DATE,
                       output=output)
    data = add_predict(active_data=data,
                       orders=extended_dataset,
                       func=apply_threshold,
                       period=INITIAL_DATE,
                       name="_snk",
                       output=output,
                       to=False,
                       threshold=100)
    data = add_predict(active_data=data,
                       orders=extended_dataset,
                       func=apply_threshold,
                       period=INITIAL_DATE,
                       name="_ber",
                       output=output,
                       to=False,
                       threshold=100)
    return data


def apply_threshold(orders, sku, area, distr, period='2019', threshold=25):
    df = interpol(orders, sku, area, distr, extra=False)

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

    if period == '2019':
        if dff[dff.index.year < int(period)].QTY.values[-3:].mean() < threshold:
            dff['pred'] = dff[dff.index.year < int(period)].QTY.values[-2:].mean()
        else:
            return None
    elif period == 'today':
        today = date.today()
        if dff[(dff.index.year < today.year - 1) |
               ((dff.index.year == today.year - 1) &
                (dff.index.month < today.month))].QTY.values[-3:].mean() < threshold:
            dff['pred'] = dff[(dff.index.year < today.year - 1) |
                              ((dff.index.year == today.year - 1) &
                               (dff.index.month < today.month))].QTY.values[-2:].mean()
        else:
            return None
    else:
        current_year = period.year
        current_month = period.month
        if dff[(dff.index.year < current_year) |
               ((dff.index.year == current_year) &
                (dff.index.month < current_month))].QTY.values[-3:].mean() < threshold:
            dff['pred'] = dff[(dff.index.year < current_year) |
                              ((dff.index.year == current_year) &
                               (dff.index.month < current_month))].QTY.values[-2:].mean()
        else:
            return None

    res = dff.reset_index()
    res["month_start"] = res['month'].values.astype('datetime64[M]')
    res = res.groupby("month_start")["QTY", "pred"].sum().reset_index()
    res["SKU"] = sku
    res["Area"] = area
    res["DISTR_NAME"] = distr
    res['month'] = res.month_start.dt.month
    res['year'] = res.month_start.dt.year
    return res


def add_predict(active_data, orders, func, period, smokie=False,
                name='', output='aaa.csv', to=True, threshold=25, show=False):
    sample_data = active_data.copy()
    if name:
        if (name == "Закарпатская") | (name == 'Undef'):
            sample_data = active_data[active_data.Area.str.contains(name)]
        else:
            sample_data = active_data[active_data.SKU.str.contains(name)]
    if smokie:
        sample_data = sample_data[~sample_data.SKU.str.contains('smokie')]
    if show:
        print('Sample_data: ', sample_data.shape)

    forecasts = []
    for sku, area, distr in sample_data[["SKU", "Area", "DISTR_NAME"]].drop_duplicates().values:
        try:
            forecasts.append(func(sample_data, sku, area, distr, period=period, threshold=threshold))
        except:
            print(sku, area, distr)
            print("Data error!")
            pass

    all_pred = pd.concat(forecasts,
                         axis=0).merge(right=orders[['Area', 'Brand', 'Status', 'Package', 'DISTR_NAME', 'QTY', 'SKU']],
                                       how='left',
                                       on=['Area', 'DISTR_NAME', 'QTY', 'SKU']
                                       ).drop_duplicates(subset=["year", "month", "SKU", "Area", "DISTR_NAME"],
                                                         keep='last')
    if show:
        print('Pred_data: ', all_pred.shape)
    if smokie:
        data = pd.concat([active_data[~active_data.SKU.str.contains(name)],
                          active_data[active_data.SKU.str.contains('smokie')],
                          all_pred], axis=0)
    else:
        data = pd.concat([active_data, all_pred],
                         axis=0).drop_duplicates(subset=["year", "month", "SKU", "Area", "DISTR_NAME"], keep='last')
    data["keg_"] = data.SKU.str.split("_").apply(lambda x: x[-1])
    if to:
        data.to_csv(output, index=False)
    return data


def result(dataset, sku, area, distr, flag=2):
    if flag == 0:
        return dataset[(dataset.SKU == sku) &
                       (dataset.Area == area) &
                       (dataset.DISTR_NAME == distr)][['SKU', 'Area', 'DISTR_NAME', 'year', 'month', 'QTY', 'pred']]
    elif flag == 1:
        return dataset[(dataset.SKU == sku) & (dataset.Area == area) & (dataset.DISTR_NAME == distr) &
                       (((dataset.year == INITIAL_DATE.year + 1) & (dataset.month < INITIAL_DATE.month)) |
                        ((dataset.year == INITIAL_DATE.year) &
                         (dataset.month >= INITIAL_DATE.month)))]
    else:
        data = dataset[(dataset.SKU == sku) & (dataset.Area == area) & (dataset.DISTR_NAME == distr) &
                       (((dataset.year == INITIAL_DATE.year + 1) & (dataset.month < INITIAL_DATE.month)) |
                        ((dataset.year == INITIAL_DATE.year) &
                         (dataset.month >= INITIAL_DATE.month)))]
        return OrderedDict(zip([':'.join(tup) for tup in (zip([str(i) for i in data.year],
                                                              [str(i) for i in data.month]))], data.pred))
