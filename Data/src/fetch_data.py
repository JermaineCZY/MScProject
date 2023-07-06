import requests

def fetch_data(api_key):
    url = f"https://api.opendota.com/api/proMatches?api_key={api_key}"
    response = requests.get(url)
    data = response.json()
    return data
