import numpy as np
import pandas as pd

def create_base(weeks=range(90), customers=range(2000)):
    
    products = range(250)

    n_weeks, n_customers, n_products = len(weeks), len(customers), len(products)

    base = pd.DataFrame({
    'week': np.array([[x] * n_products * n_customers for x in weeks]).flatten(),
    'customer': np.array([[x] * n_products for x in customers] * n_weeks).flatten(),
    'product': list(range(n_products)) * n_customers * n_weeks
    })
    
    return base

def add_basket_info(base, baskets):
    base = pd.merge(base, baskets, on=['week', 'customer','product'], how='left')
    base['price'] = base['price'].fillna(0).astype(int)
    base['isBought'] = (base['price'] > 0)
    return base
    
def add_coupon_info(base, baskets, coupons):
    base = pd.merge(base, coupons, on=['week', 'customer','product'], how='left')
    base['discount'] = base['discount'].fillna(0).astype(int)
    base = base.rename(columns={"discount": "dGiven"})
    base['isGiven'] = (base['dGiven'] > 0)

    normal_prices = baskets.groupby('product')['price'].max().values
    base['highestPrice'] = base['product'].apply(lambda x: normal_prices[x])
    base['isUsed'] = ((base['price'] != base['highestPrice']) & (base['price']!=0))
    base.drop('highestPrice', axis=1, inplace=True)
    return base


def get_baskets():   
    datasets_path = lambda  file_name: f'/Users/stijnvanleeuwen/Desktop/codes/EUR/Ass2/datasets/{file_name}.parquet'
    return pd.read_parquet(datasets_path('baskets')).astype({'week':'uint8', 'customer':'uint','product':'category', 'price':'uint16'}) 

def get_coupons():
    datasets_path = lambda  file_name: f'/Users/stijnvanleeuwen/Desktop/codes/EUR/Ass2/datasets/{file_name}.parquet'
    return pd.read_parquet(datasets_path('coupons')).astype({'week':'uint8', 'customer':'uint','product':'category', 'discount':'uint8'})
    
def get_prediction_index():
    datasets_path = lambda  file_name: f'/Users/stijnvanleeuwen/Desktop/codes/EUR/Ass2/datasets/{file_name}.parquet'    
    return pd.read_parquet(datasets_path('prediction_index')).astype({'week':'uint8', 'customer':'category','product':'category'}) 

def get_3_files():
    return get_baskets(), get_coupons(), get_prediction_index()
        
def split_4_way(base, target_col, unkwown_week=89):
    train = base[base['week']!=unkwown_week]
    test =  base[base['week']==unkwown_week]
    x_train, x_test = train.drop(target_col,axis=1), test.drop(target_col,axis=1)
    y_train, y_test = train[target_col], test[target_col]
    return x_train, y_train, x_test, y_test 