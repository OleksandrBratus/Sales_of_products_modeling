import pymssql
import numpy as np
import pandas as pd
from config.config import mssql_config


def load_orders(to='data/orders.csv'):
    orders = load_df(orders_query)
    renames = {
        "LIMONADIYA_50_keg_Grapefruit_ban": "lymonadiya_50_keg_grapefruit_ban",
        "LIMONADIYA_50_keg_Orange_ban": "lymonadiya_50_keg_orange_ban"
    }
    orders.SKU = orders.SKU.replace(renames)
    orders.to_csv(to, index=False)
    return orders


def load_products(to='data/products.csv', info='data/INFO_SKU.xlsx', repl='data/replace_sku.xlsx'):
    products = load_df(products_query).drop(["DLM"], axis=1)
    products.SKU = products.SKU.str.lower()
    products.ProductVolume = products.ProductVolume.astype("float")
    replace = pd.read_excel(info)
    replace.columns = ["SKU", "SKU_NEW", "Status"]
    for column in replace.columns:
        replace[column] = replace[column].str.lower()
    products = products.merge(replace, how="outer", on="SKU")
    products.SKU = products.SKU_NEW.values
    products.drop(["SKU_NEW"], axis=1, inplace=True)
    replace_columns = ["SKU_DESCRIPTION", "SKU_NEW"]
    replace = pd.read_excel(repl)[replace_columns]
    replace.columns = ["SKU", "SKU_NEW"]
    for column in replace.columns:
        replace[column] = replace[column].str.lower()
    replace = replace[replace.SKU_NEW != "старе скю"]
    replace.drop_duplicates(inplace=True)

    exceptions = [
        "BEERTAIL_50_keg_5,0%_Plum wine_san",
        "BEERTAIL_50_keg_5,0%_Aperol spritz_ber",
        "Lymonadiya_50_keg_Orange_ban",
        "Lymonadiya_50_keg_Grapefruit_ban"
    ]
    exceptions = list(map(lambda x: str(x).lower(), exceptions))

    new_products = products[["SKU"]].merge(replace, how="outer", on="SKU")
    new_products.SKU_NEW = np.where(new_products.SKU_NEW.notna(), new_products.SKU_NEW, new_products.SKU)
    new_products.drop_duplicates(inplace=True)
    new_products = new_products[new_products.SKU_NEW.isin(replace.SKU_NEW.tolist() + exceptions)]
    new_products = new_products.merge(products, left_on="SKU_NEW", right_on="SKU", how="left", suffixes=('', 'old'))
    new_products.drop(["SKUold"], axis=1, inplace=True)
    new_products.drop_duplicates(inplace=True)
    new_products.to_csv(to, index=False)
    return new_products


def load_df(query, config=mssql_config, dates=None):
    conn = pymssql.connect(**config)
    df = pd.read_sql(query, conn, parse_dates=dates)
    conn.close()
    return df


orders_query = """
SELECT
    YEAR(IP_tblIWIS_orders.Date_ID) as year,
    MONTH(IP_tblIWIS_orders.Date_ID) as month,
    SUM(QTY) as QTY,
    SKU,
    Area,
    DISTR_NAME,
    SUM(TOTAL_AMOUNT) as TOTAL_AMOUNT
FROM IP_tblIWIS_orders
INNER JOIN IP_tblIWIS_TT ItIT on IP_tblIWIS_orders.Ol_ID = ItIT.Ol_ID
GROUP BY
    SKU,
    Area,
    DISTR_NAME,
    YEAR(IP_tblIWIS_orders.Date_ID),
    MONTH(IP_tblIWIS_orders.Date_ID)
ORDER BY
    SKU,
    Area,
    DISTR_NAME,
    YEAR(IP_tblIWIS_orders.Date_ID),
    MONTH(IP_tblIWIS_orders.Date_ID)
"""


products_query = """
SELECT *
FROM [SWE_Transit].[dbo].[IP_tblIWIS_products]
"""
