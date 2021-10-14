import numpy as np
import pandas as pd

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