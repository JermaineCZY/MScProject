import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from ast import literal_eval


def process_purchase_log(purchase_log):
    if pd.isna(purchase_log):
        purchase_log = '[]'
    purchase_log_list = literal_eval(purchase_log)
    items = []

    for item in purchase_log_list:
        items.append(item['key'])

    return pd.Series([items])


# read the csv file
df = pd.read_csv('../raw/match_details.csv')

df_temp = df.copy()  # create a temporary data frame
for col in df_temp.columns:
    if 'purchase_log' in col:
        df_temp[f'{col}_full'] = df_temp[col].apply(lambda x: process_purchase_log(x))
        df_temp = df_temp.drop(col, axis=1)  # remove the original purchase column

# collect all purchased items
all_items = set()
for col in df_temp.columns:
    if 'purchase_log' in col:
        for items in df_temp[col]:
            all_items.update(items)

all_items = list(all_items)

# one hot encoding of all purchases
for col in df_temp.columns:
    if 'purchase_log' in col:
        item_onehot = []
        for i in range(len(df_temp)):
            row = [0] * len(all_items)
            for item in df_temp[col][i]:
                if item in all_items:
                    index = all_items.index(item)
                    row[index] = 1
            item_onehot.append(row)
        item_onehot = np.array(item_onehot)
        df_temp = pd.concat([df_temp, pd.DataFrame(item_onehot, columns=all_items)], axis=1)
        df_temp = df_temp.drop(col, axis=1)  # remove the original purchase column

# one hot encoding of all heroes
heroes_cols = [col for col in df_temp.columns if 'hero' in col]
enc_hero = OneHotEncoder(dtype=int)
for col in heroes_cols:
    onehot_results = enc_hero.fit_transform(df_temp[[col]]).toarray()
    df_temp = pd.concat([df_temp, pd.DataFrame(onehot_results)], axis=1)
    df_temp = df_temp.drop(col, axis=1)  # remove the original hero column

# Convert true and false in the radiant win column to 1 and 0
df_temp['radiant_win'] = df_temp['radiant_win'].map({True: 1, False: 0})

# save the processed data to a new csv file
df_temp.to_csv(f'processed_data_full.csv', index=False)
