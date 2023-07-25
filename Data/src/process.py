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


# 读取csv文件
df = pd.read_csv('../raw/match_details.csv')

df_temp = df.copy()  # 创建一个临时的DataFrame，以便我们可以在原始数据上进行操作
for col in df_temp.columns:
    if 'purchase_log' in col:
        df_temp[f'{col}_full'] = df_temp[col].apply(lambda x: process_purchase_log(x))
        df_temp = df_temp.drop(col, axis=1)  # 移除原有的购物记录列

# 收集所有购买的物品
all_items = set()
for col in df_temp.columns:
    if 'purchase_log' in col:
        for items in df_temp[col]:
            all_items.update(items)

all_items = list(all_items)

# 对所有的购物记录进行one-hot编码
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
        df_temp = df_temp.drop(col, axis=1)  # 移除原有的购物记录列

# 对所有的hero进行one-hot编码
heroes_cols = [col for col in df_temp.columns if 'hero' in col]
enc_hero = OneHotEncoder(dtype=int)
for col in heroes_cols:
    onehot_results = enc_hero.fit_transform(df_temp[[col]]).toarray()
    df_temp = pd.concat([df_temp, pd.DataFrame(onehot_results)], axis=1)
    df_temp = df_temp.drop(col, axis=1)  # 移除原有的hero列

# 将radiant_win列中的true和false转化为1和0
df_temp['radiant_win'] = df_temp['radiant_win'].map({True: 1, False: 0})

# 将处理后的数据保存到新的CSV文件
df_temp.to_csv(f'processed_data_full.csv', index=False)
