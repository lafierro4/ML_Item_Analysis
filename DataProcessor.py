import pandas as pd
import os

#Loading Dataset
item_data = pd.read_csv('Item Data/lol_items_stats.csv', sep=',')

#Transfroming percentage based stats into decimal, aka 0.12 -> 12, to standerize stats
def percentage_to_decimal(dataframe, columns):
    dataframe[columns] = dataframe[columns] * 100

percentage_to_decimal(item_data, ['AS','Crit','LS','APen','MP5','HSP','OVamp','MPen','HP5'])

dataitemstat = item_data[['AD','AS','Crit','LS','APen','AP','AH','Mana','MP5','HSP','OVamp','MPen','Health','Armor','MR','HP5','MS']]
#print(dataitemstat)


itemcost = item_data[['Item','Cost']]
#print(itemcost)

sumstat=dataitemstat.sum(axis=1)
#print(sumstat)

goldstat = pd.concat([itemcost,sumstat], axis=1)
#print(goldstat)

goldstat=goldstat.rename(columns = {0: 'Sumstat'})
#print(goldstat)

ratiocal = pd.DataFrame((goldstat.Sumstat/goldstat.Cost)*100)
StatGoldRatio = pd.concat([goldstat,ratiocal], axis=1)
StatGoldRatio = StatGoldRatio.sort_values(by=0)
StatGoldRatio=StatGoldRatio.rename(columns = {0: 'Stats/Gold ratio'})
#print(StatGoldRatio)

# Step 1: Calculate the total value of the stats provided by each item
item_data['TotalStatsValue'] = item_data[['AD','AS','Crit','LS','APen','AP','AH','Mana','MP5','HSP','OVamp','MPen','Health','Armor','MR','HP5','MS']].sum(axis=1)

# Step 2: Calculate the gold efficiency of each item
item_data['GoldEfficiency'] = item_data['TotalStatsValue'] / item_data['Cost']

# Step 3: Print the item name and its gold efficiency
item_efficiency = item_data[['Item', 'GoldEfficiency']]
#print(item_efficiency)


print(item_data)