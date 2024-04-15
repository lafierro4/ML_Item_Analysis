import requests
import json
import pandas as pd
import DataProcessor 
version = "14.7.1" #change to get specific version data, follow 14.x.1 guide, latest available version 14.7.1
language = "en_US"

#connecting to the the api
try:
    data_url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/{language}/item.json"
    data_response = requests.get(data_url)
    print(data_response.status_code)
except ConnectionError as error:
    print("Connection Error: ", error) 

#getting the json from the response
items_data = data_response.json()

# seperate items, only getting items that are usuable to the project and get all the needed individual item infomation  
items_list = []
if "data" in items_data:
    for item_id, item in items_data["data"].items():
        if all([not item.get(key, False) for key in ["requiredChampion", "requiredAlly"]]) and \
           item.get("maps", {}).get("11", False) and \
           item.get("gold", {}).get("purchasable", False) and \
           not item.get("hideFromAll", False):
            stats = item.get("stats", {})
            if stats: #if the item doesnt have statWs skip it,
                items_list.append({
                    "item_id": item_id,
                    "name": item.get("name", ""),
                    "gold": item.get("gold", {}),
                    "stats": stats
                })
#making json file with all of the item's infomation for refernce
with open(f"Item Data/{version}_item_data.json", 'w') as data_file:
    output_data = {"item_count": len(items_list), "items": items_list}
    json.dump(output_data, data_file, indent=2)

#making a csv file from the json file that has the seperated items
# headers from item_data
headers = ["item_id", "name", "total_gold"]
for stat in items_data['basic']['stats']:
        headers.append(stat)

data_rows = []
for item in items_list:
    item_info = [item['item_id'], item['name'], item['gold'].get('total', 0)]
    for stat in items_data['basic']['stats']:
        item_info.append(item['stats'].get(stat, 0))  #set the stat to 0 if the item does not have an associated value for stat
    data_rows.append(item_info)

df = pd.DataFrame(data_rows, columns=headers)
df.to_csv(f"Item Data/{version}_item_data.csv", index=False)
