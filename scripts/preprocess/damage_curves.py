"""Process asset information and write final resutls to geopackages
"""
import os
import sys
import pandas as pd
import numpy as np
from preprocess_utils import *

def assign_damage_percent(x,flood_depth,asset_type,multiplication_factor=1):
    """Inputs:
        x: Pandas Dataframe object that should have the 
        flood_depth: String name of flood depth column
        asset_type: String name of asset class column
    """
    flood_depth = float(x[flood_depth])
    if x[asset_type] == 'Power Plant':
        flood_depth = 3.2808*flood_depth
        if flood_depth <= 8:
            damage_percent = 2.5*flood_depth
        else:
            damage_percent = 5.0*flood_depth - 20.0
    elif x[asset_type] == 'Reservoir':
        flood_depth = 3.2808*flood_depth
        if flood_depth <= 0.5:
            damage_percent = 5.0*flood_depth
        else:
            damage_percent = 180.0*flood_depth - 170.0
    elif x[asset_type] == 'Road':
        if flood_depth <= 0.5:
            damage_percent = 40.0*flood_depth
        elif 0.5 < flood_depth <= 1.0:
            damage_percent = 30.0*flood_depth + 5.0
        else:
            damage_percent = 10.0*flood_depth + 25.0
    elif x[asset_type] == 'Airport':
        # Source: Kok, M., Huizinga, H.J., Vrouwenvelder, A. and Barendregt, A. (2004), 
        # “Standard method 2004 damage and casualties caused by flooding”.
        damage_percent = 100.0*min(flood_depth,0.24*flood_depth+0.4,0.07*flood_depth+0.75,1)
    elif x[asset_type] in ('Landfill','WWTP'):
        if flood_depth <= 0:
            damage_percent = 0
        elif 0 < flood_depth <= 0.5:
            damage_percent = 4
        elif 0.5 < flood_depth <= 1.0:
            damage_percent = 8
        elif 1.0 < flood_depth <= 2.0:
            damage_percent = 17
        elif 2.0 < flood_depth <= 3.0:
            damage_percent = 25    
        else:
            damage_percent = 30

    damage_percent = multiplication_factor*damage_percent
    if damage_percent > 100:
        damage_percent = 100

    return damage_percent/100.0

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    sector_descriptions = [
                        {
                            'sector':'landfill',
                            'asset_type':'Landfill'
                        },
                        {
                            'sector':'air',
                            'asset_type':'Airport'
                        },
                        {
                            'sector':'power',
                            'asset_type':'Power Plant'
                        },
                        {
                            'sector':'road',
                            'asset_type':'Road'
                        }
                        ]
    damage_excel = os.path.join(processed_data_path,"damage_curves",'damage_curves_flooding.xlsx')
    excl_wrtr = pd.ExcelWriter(damage_excel)

    for sector in sector_descriptions:
        df = pd.DataFrame()
        df["flood_depth"] = list(np.linspace(0,10,20))
        df["asset_type"] = sector["asset_type"]
        df["min_damage_ratio"] = df.apply(lambda x: assign_damage_percent(x,"flood_depth","asset_type",multiplication_factor=1),axis=1)
        df["max_damage_ratio"] = df.apply(lambda x: assign_damage_percent(x,"flood_depth","asset_type",multiplication_factor=5),axis=1)

        df[["flood_depth","min_damage_ratio","max_damage_ratio"]].to_excel(excl_wrtr, sector["sector"], index=False)
    
    excl_wrtr.save()



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
