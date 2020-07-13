import numpy as np
import pandas as pd
from datetime import date
import lightgbm as lgb
from config.config import MAX_DATE, INITIAL_DATE


class LightModel:

    def __init__(self, data):
        self.models = []
        self.dataset = data

    def train(self):
        for type in list(np.unique(self.dataset.SKU.str.split("_").apply(lambda x: x[-1]))):
            final_df = self.dataset[self.dataset.SKU.str.contains(f"_{type}")]
            self.models.append(self._train(self, final_df=final_df, type_=type+"_", period=INITIAL_DATE))

    def predict(self, output):
        data = []
        for type_, model in zip(list(np.unique(self.dataset.SKU.str.split("_").apply(lambda x: x[-1]))), self.models):
            final_df = self.dataset[self.dataset.SKU.str.contains(f"_{type_}")]
            final_df["pred"] = np.maximum(0, model.predict(self._to_cat(final_df.drop(["QTY"], axis=1))))
            data.append(final_df)

        data = pd.concat(data)
        data = data.drop_duplicates(subset=["year", "month", "SKU", "Area", "DISTR_NAME"], keep='last')
        data.to_csv(output, index=False)
        return data

    @staticmethod
    def _to_cat(final_df):
        categorical = ["Area", "DISTR_NAME", "Package", "Brand", "Category", "SKU", "Status", "keg_"]
        for column in categorical:
            final_df[column] = final_df[column].astype("category")
        return final_df

    @staticmethod
    def _split(self, final_df, period='2019'):
        final_df = self._to_cat(final_df)

        if period == '2019':
            train_df = final_df[(final_df.year < 2019)]
            test_df = final_df[final_df.year == 2019]
        elif period == 'today':
            today = date.today()
            train_df = final_df[(final_df.year < today.year - 1) |
                                ((final_df.year == today.year - 1) & (final_df.month < today.month))]
            test_df = final_df[(final_df.year > today.year - 1) |
                               ((final_df.year == today.year - 1) & (final_df.month >= today.month))]
        else:
            current_year = period.year
            current_month = period.month
            train_df = final_df[(final_df.year < current_year) |
                                ((final_df.year == current_year) & (final_df.month < current_month))]
            test_df = final_df[((final_df.year > current_year) |
                                ((final_df.year == current_year) & (final_df.month >= current_month))) &
                               (((final_df.year == MAX_DATE.year) & (final_df.month <= MAX_DATE.month)) |
                                (final_df.year < MAX_DATE.year))]

        X_train = train_df.drop(["QTY"], axis=1)
        Y_train = train_df["QTY"]

        X_test = test_df.drop(["QTY"], axis=1)
        Y_test = test_df["QTY"]
        return X_train, X_test, Y_train, Y_test, train_df, test_df

    @staticmethod
    def _train(self, final_df, type_, period='2019'):

        X_train, X_test, Y_train, Y_test, train_df, test_df = self._split(self, final_df, period=period)

        params = {
            'objective': 'regression_l1',
            'num_leaves': 100,
            'max_depth': 9,
            'learning_rate': 0.01,
            ""

            'min_data_in_leaf': 4,
            'bagging_fraction': 1.0,

            'feature_fraction': 1.0,

            'lambda_l1': 0.1,
            'lambda_l2': 0.1,

            'min_gain_to_split': 0,
            'metric': 'mae',

            'n_estimators': 10000,
            'random_state': 42,
            'n_jobs': -1,
            "verbose": 1
        }

        lgbm = lgb.LGBMRegressor(**params)

        fNames = X_train.columns.tolist()

        lgbm.fit(X_train,
                 Y_train,
                 eval_metric="mae",
                 verbose=0,
                 eval_set=(X_test, Y_test),
                 early_stopping_rounds=20,
                 feature_name=fNames)
        final_df["pred"] = lgbm.predict(final_df.drop(["QTY"], axis=1))
        # final_df.to_csv(f"{type_}To_BPI_2020_01_29.csv", index=False)
        return lgbm
