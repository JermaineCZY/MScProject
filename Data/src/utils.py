import datetime
import requests
import json
import pandas as pd
import os

# timestamp to date
def timestamp_to_date(timestamp):
    date = datetime.datetime.fromtimestamp(timestamp)
    print(date)


def test():
    # sql query
    sql_query = """SELECT DISTINCT
        matches.match_id,
        matches.start_time,
        leagues.name leaguename
        FROM matches
        JOIN match_patch using(match_id)
        JOIN leagues using(leagueid)
        JOIN player_matches using(match_id)
        WHERE TRUE
        AND matches.start_time >= extract(epoch from timestamp '2023-04-19T23:00:00.000Z')
        AND matches.start_time <= extract(epoch from timestamp '2023-06-29T00:00:00.000Z')
        AND (leagues.tier = 'professional' OR leagues.tier = 'premium')
                ORDER BY matches.match_id NULLS LAST
        LIMIT 10000
    """

    response = requests.get(
        "https://api.opendota.com/api/explorer",
        params={"sql": sql_query}
    )

    data = response.json()
    # create a pandas dataframe
    df = pd.DataFrame(data['rows'])

    # save data frame as a csv file
    df.to_csv('output.csv', index=False)
    # print query results
    # print(json.dumps(data, indent=4))

def split_matches_csv(file_path):

    # 读取CSV文件
    path = file_path
    df = pd.read_csv(path)

    # 获取原始文件名（不包括扩展名）
    original_filename = os.path.splitext(os.path.basename(path))[0]

    # 打乱数据
    df_shuffled = df.sample(frac=1, random_state=42)

    # 划分数据集
    train_size = int(0.8 * len(df_shuffled))  # 例如，训练集占80%
    train_set = df_shuffled[:train_size]
    validation_set = df_shuffled[train_size:]

    # 保存为新的CSV文件
    train_set.to_csv(f'{original_filename}_train_set.csv', index=False)
    validation_set.to_csv(f'{original_filename}_validation_set.csv', index=False)


if __name__ == '__main__':
    split_matches_csv('processed_data_25_min.csv')
    # timestamp_to_date(1687994215)
