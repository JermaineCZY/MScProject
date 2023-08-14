import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from ast import literal_eval


def process_purchase_log(purchase_log, time_limit):
    if pd.isna(purchase_log):
        purchase_log = '[]'
    purchase_log_list = literal_eval(purchase_log)
    items_within_time_limit = []

    for item in purchase_log_list:
        if item['time'] <= time_limit:
            items_within_time_limit.append(item['key'])

    return pd.Series([items_within_time_limit])


# read csv file
df = pd.read_csv('../raw/match_details.csv')

# Process all the shopping records and add new columns, then save them to different CSV files.
for time_limit in [300, 600, 900, 1200, 1500, 1800, 2100]:  # 5 minutes, 10 minutes, 15 minutes, 20 minutes, 25 minutes, 30 minutes, 35 minutes.
    df_temp = df.copy()  # Create a temporary Data Frame so that we can perform operations on the original data.
    for col in df_temp.columns:
        if 'purchase_log' in col:
            df_temp[f'{col}_{time_limit//60}_min'] = df_temp[col].apply(lambda x: process_purchase_log(x, time_limit))
            df_temp = df_temp.drop(col, axis=1)  # remove the existing shopping record column

    # collect all purchased items
    all_items = set()
    for col in df_temp.columns:
        if 'purchase_log' in col:
            for items in df_temp[col]:
                all_items.update(items)

    all_items = list(all_items)

    # perform one hot encoding on all shopping records
    for col in df_temp.columns:
        if 'purchase_log' in col:
            item_onehot = []
            for i in range(len(df_temp)):
                row = [0] * len(all_items)
                for item in df_temp[col][i]:
                    if item in all_items:
                        index = all_items.index(item)
                        row[index] = 2
                item_onehot.append(row)
            item_onehot = np.array(item_onehot)
            df_temp = pd.concat([df_temp, pd.DataFrame(item_onehot, columns=all_items)], axis=1)
            df_temp = df_temp.drop(col, axis=1)  # remove the existing shopping record column

    # one hot encode all the heroes
    heroes_cols = [col for col in df_temp.columns if 'hero' in col]
    enc_hero = OneHotEncoder(dtype=int)
    for col in heroes_cols:
        onehot_results = enc_hero.fit_transform(df_temp[[col]]).toarray()
        df_temp = pd.concat([df_temp, pd.DataFrame(onehot_results)], axis=1)
        df_temp = df_temp.drop(col, axis=1)  # remove the original column of heroes

    # Convert the 'true' and 'false' values in the 'radiant win' column to '1' and '0'.
    df_temp['radiant_win'] = df_temp['radiant_win'].map({True: 1, False: 0})

    # save the processed data to a new csv file
    df_temp.to_csv(f'processed_data_{time_limit//60}_min_new.csv', index=False)
