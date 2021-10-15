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
    """
    Returns baskets, coupons, prediction_index
    """
    return get_baskets(), get_coupons(), get_prediction_index()
        
def split_4_way(base, target_col, unkwown_week=89):
    train = base[base['week']!=unkwown_week]
    test =  base[base['week']==unkwown_week]
    x_train, x_test = train.drop(target_col,axis=1), test.drop(target_col,axis=1)
    y_train, y_test = train[target_col], test[target_col]
    return x_train, y_train, x_test, y_test 


# ALLLLLL

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
    base['basket'] = base['week'].astype(str) + '_' + base['customer'].astype(str)
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

def add_categories(base):
    base['category'] = (base['product'] / 10).astype(int)
    return base

def add_frequencies(base, baskets):
    
    baskets.loc[:,'category'] = (baskets['product'].astype(int) / 10).astype(int)
    n_weeks = baskets['week'].nunique()
    
    prod_probs_df = (
        baskets.groupby(['customer','product'])['week'].count() / n_weeks) \
        .reset_index() \
        .rename(columns={'week':'probability'})

    cat_probs_df = (
        baskets.groupby(['customer','category'])['week'].count() / n_weeks) \
        .reset_index() \
        .rename(columns={'week':'probability'})
        
    base = pd.merge(base, prod_probs_df, on=['customer','product'] ,how='left')
    base = pd.merge(base, cat_probs_df, on=['customer','category'] ,how='left')
    
    base.rename(columns={'probability_x':'p_prod', 'probability_y':'p_cat'},inplace=True)
    
    base[['p_prod','p_cat']] = base[['p_prod','p_cat']].fillna(0)
    
    return base

def get_rolling_frequencies(base, n_weeks=5, category=False):
    
    rolling_df = pd.DataFrame()
    
    for week_nr in range(n_weeks,89+1):

        start = week_nr - n_weeks - 1
        end = week_nr

        single_week = (
            base[(start < base['week']) & (base['week'] < end)] 
            .groupby(['customer',f"{'category' if category else 'product'}"]) 
            .agg({'week':'last','isBought':'sum'}) 
            .reset_index()
            )
        
        single_week['week'] = single_week['week'] + 1
        single_week['isBought'] = single_week['isBought'] / n_weeks
        rolling_df = pd.concat([rolling_df, single_week])
    
    return rolling_df

def add_rolling_frequencies(base):
    
    values = [5,10,30]
    prod_names = [f'roll_prod_{value}' for value in values]
    cat_names  = [f'roll_cat_{value}' for value in values]

    for i, name in enumerate(prod_names):
        rolled = get_rolling_frequencies(base,n_weeks=values[i]).rename(columns={'isBought':name})
        base = pd.merge(base, rolled, on=['week','customer','product'],how='left')

    for i, name in enumerate(cat_names):
        rolled = get_rolling_frequencies(base, n_weeks=values[i], category=True).rename(columns={'isBought':name})
        base2 = pd.merge(base, rolled, on=['week','customer','category'],how='left') 
        
    base.loc[:,prod_names[0]:] = base2.loc[:,prod_names[0]:].fillna(0)  
    
    return base

def buy_weeks_to_ago(buy_weeks):
    weeks_past = 90
    val = []
    bought = False

    for i in range(90):
        if i in buy_weeks:
            weeks_past = 0
            val.append(weeks_past)
            bought = True
        elif(bought==True): 
            weeks_past += 1 
            val.append(weeks_past)
        else:
            val.append(90)
    return val


def add_weeks_ago(base):
    
    only_bought = base[base['isBought']==1]
    pairs = only_bought.groupby(['customer','product']).size().reset_index(inplace=False)

    all_pairs = pd.DataFrame()

    for i in range(len(pairs)):
        customer = pairs.iloc[i,0]
        product = pairs.iloc[i,1]
        
        df = pd.DataFrame({
            'week':range(90),
            'customer':customer, 
            'product':product, 
            })
        
        buy_weeks = list(only_bought[(only_bought['product']==product) & (only_bought['customer']==customer)]['week'])
        df['weeks_ago'] = buy_weeks_to_ago(buy_weeks)
        all_pairs = pd.concat([all_pairs,df])
        
    base = pd.merge(base, all_pairs, on=['week','customer','product'],how='left').fillna(90)
    
    return base

