import json
from fetch_data import get_matches_csv, get_matches_ids, fetch_match_details
from process_data import select_heroes, one_hot_encoding

if __name__ == '__main__':
    # load api keys
    with open("../config/api_key.json", "r") as file:
        api_keys = json.load(file)
        opendota_api_key = api_keys["opendota"]

    # get data
    # get_matches_csv('2023-04-19T23:00:00.000Z', '2023-06-29T00:00:00.000Z')
    match_ids = get_matches_ids('2023-04-19T23:00:00.000Z', '2023-06-29T00:00:00.000Z')
    # fetch_match_details(match_ids, opendota_api_key)

    # process data
    # select_heroes('../raw/match_details.csv')
    one_hot_encoding('../processed/selected_match_details.csv')
