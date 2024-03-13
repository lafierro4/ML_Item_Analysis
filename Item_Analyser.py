#main file to train Algorithm
import os
import pandas as pd
import numpy as np


#this version was the first to release this year, and has the item stats at their purest level, could be used the training set or test set
base_dataset = pd.read_csv(os.path.join("Item Data","14.1.1_item_data.csv")) 
latest_dataset = pd.read_csv(os.path.join("Item Data","14.5.1_item_data.csv")) #latest version, used for verification testing
#other versions can be used as training


