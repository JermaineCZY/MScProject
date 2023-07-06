import time
import requests
import pandas as pd
from datetime import datetime


def get_matches_csv(start_time, end_time):
    # SQL query to get matches
    sql_query = f"""SELECT DISTINCT
        matches.match_id,
        matches.start_time,
        leagues.name leaguename
        FROM matches
        JOIN match_patch using(match_id)
        JOIN leagues using(leagueid)
        JOIN player_matches using(match_id)
        WHERE TRUE
        AND matches.start_time >= extract(epoch from timestamp '{start_time}')
        AND matches.start_time <= extract(epoch from timestamp '{end_time}')
        AND (leagues.tier = 'professional' OR leagues.tier = 'premium')
                ORDER BY matches.match_id NULLS LAST
        LIMIT 10000
    """

    response = requests.get(
        "https://api.opendota.com/api/explorer",
        params={"sql": sql_query}
    )

    data = response.json()
    # 创建一个pandas DataFrame
    df = pd.DataFrame(data['rows'])

    # 从开始和结束时间中提取月和日
    start_month_day = datetime.fromisoformat(start_time.replace("Z", "+00:00")).strftime('%m%d')
    end_month_day = datetime.fromisoformat(end_time.replace("Z", "+00:00")).strftime('%m%d')

    # 将DataFrame保存为CSV文件，文件名包含开始和结束的月和日
    df.to_csv(f'matchids_{start_month_day}_{end_month_day}.csv', index=False)


def get_matches_ids(start_time, end_time):
    # SQL query to get matches
    sql_query = f"""
    SELECT DISTINCT
        matches.match_id
    FROM matches
    JOIN match_patch using(match_id)
    JOIN leagues using(leagueid)
    JOIN player_matches using(match_id)
    WHERE TRUE
    AND matches.start_time >= extract(epoch from timestamp '{start_time}')
    AND matches.start_time <= extract(epoch from timestamp '{end_time}')
    AND (leagues.tier = 'professional' OR leagues.tier = 'premium')
    ORDER BY matches.match_id NULLS LAST
    LIMIT 10000
    """

    response = requests.get(
        "https://api.opendota.com/api/explorer",
        params={"sql": sql_query}
    )

    data = response.json()

    # 创建一个只包含match_id的Python列表
    match_ids = [row['match_id'] for row in data['rows']]
    return match_ids


def fetch_match_details(match_ids):
    match_details = []
    for match_id in match_ids:
        response = requests.get(f"https://api.opendota.com/api/matches/{match_id}?api_key=95c09c36-f3bf-4748-ac5a-604be59a8bdd")
        data = response.json()

        match_info = {"match_id": match_id, "start_time": data["start_time"]}
        players = sorted(data["players"], key=lambda x: x["player_slot"])

        for i, player in enumerate(players):
            team = "radiant" if player["player_slot"] < 128 else "dire"
            match_info[f"{team}_hero_{i % 5 + 1}"] = player["hero_id"]
            match_info[f"{team}_purchase_log_{i % 5 + 1}"] = player.get("purchase_log")

        match_info["radiant_win"] = data["radiant_win"]
        match_details.append(match_info)

        time.sleep(60 / 1000)   # 1000 requests per minute
    df = pd.DataFrame(match_details)
    df.to_csv("match_details.csv", index=False)
    # return match_details


if __name__ == '__main__':
    # get_matches_csv('2023-04-19T23:00:00.000Z', '2023-06-29T00:00:00.000Z')
    match_ids = get_matches_ids('2023-04-19T23:00:00.000Z', '2023-06-29T00:00:00.000Z')
    fetch_match_details(match_ids)
