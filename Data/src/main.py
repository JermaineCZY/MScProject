import json
from fetch_data import fetch_data
from process_data import select_heroes
from save_data import save_data

# 加载API keys
with open("../config/api_key.json", "r") as file:
    api_keys = json.load(file)
    opendota_api_key = api_keys["opendota"]

# 获取数据
raw_data = fetch_data(opendota_api_key)

# 处理数据
processed_data = select_heroes(raw_data)

# 保存数据
save_data(processed_data, "data/processed/data.csv")
