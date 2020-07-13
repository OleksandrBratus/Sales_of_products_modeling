import pandas as pd


def load_datasets(path='./../data/'):
    orders = pd.read_csv(path+"orders.csv")
    products = pd.read_csv(path+"products.csv")
    orders.SKU = orders.SKU.str.lower()
    replace = pd.read_excel(path+"INFO_SKU.xlsx")
    replace.columns = ["SKU", "SKU_NEW", "Status"]
    for column in replace.columns:
        replace[column] = replace[column].str.lower()
    replace = replace[replace.SKU != replace.SKU_NEW]
    orders.SKU = orders.SKU.replace(replace.set_index("SKU")["SKU_NEW"].to_dict()).values
    orders = orders.merge(products[["SKU", "SKU_NEW"]], on="SKU", how="inner")
    orders.SKU = orders.SKU_NEW.values
    orders.drop(["SKU_NEW", "TOTAL_AMOUNT"], axis=1, inplace=True)
    orders = orders[orders.SKU.notna()]
    new_sku = orders.groupby("SKU").year.min()[orders.groupby("SKU").year.min() < 2019].index
    orders = orders[orders.SKU.isin(new_sku)]
    orders = pd.DataFrame(orders.groupby(by=['year', 'month', 'SKU', 'Area', 'DISTR_NAME']).QTY.sum().reset_index())[
        ['year', 'month', 'QTY', 'SKU', 'Area', 'DISTR_NAME']]
    print(orders.shape)

    return orders, products