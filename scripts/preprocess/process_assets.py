"""Process asset information and write final resutls to geopackages
"""
import os
import sys
import pandas as pd
import geopandas as gpd
from preprocess_utils import *

def damage_costs_and_economic_loss_estimates(asset_df,asset_dictionary,cost_uncertainty_factor=0.25,loss_uncertainty_factor=0.25):
    if asset_dictionary['length_column'] is not None:
        asset_df['mean_damage_cost'] = (1.0*asset_dictionary['cost_conversion']/asset_dictionary['length_unit'])*asset_df[asset_dictionary['cost_column']]
    else:
        asset_df['mean_damage_cost'] = asset_dictionary['cost_conversion']*asset_df[asset_dictionary['cost_column']]

    asset_df['min_damage_cost'] = (1 - cost_uncertainty_factor)*asset_df['mean_damage_cost']
    asset_df['max_damage_cost'] = (1 + cost_uncertainty_factor)*asset_df['mean_damage_cost']
        
    #################################################################################################
    # The economic losses are in RMB/day
    #################################################################################################
    if asset_dictionary['min_economic_loss_column'] != asset_dictionary['max_economic_loss_column']: 
        asset_df['min_economic_loss'] = asset_dictionary['economic_loss_conversion']*asset_df[asset_dictionary['min_economic_loss_column']].sum(axis=1)
        asset_df['max_economic_loss'] = asset_dictionary['economic_loss_conversion']*asset_df[asset_dictionary['max_economic_loss_column']].sum(axis=1)
    else:
        asset_df['economic_loss'] = asset_dictionary['economic_loss_conversion']*asset_df[asset_dictionary['min_economic_loss_column']].sum(axis=1)
        asset_df['min_economic_loss'] = (1 - loss_uncertainty_factor)*asset_df['economic_loss']
        asset_df['max_economic_loss'] = (1 + loss_uncertainty_factor)*asset_df['economic_loss']

    asset_df['damage_cost_unit'] = asset_dictionary['damage_cost_unit']
    asset_df['economic_loss_unit'] = asset_dictionary['economic_loss_unit']

    return asset_df

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    china_proj = 4326
    sector_descriptions = [
                        {
                            'sector':'landfill',
                            'sector_type':'nodes',
                            'sector_name':'Landfill',
                            'id_column':'Plant_Numb',
                            'adaptation_criteria_column':'Capacity_1',
                            'flood_protection_column':'Flood_protect',
                            'cost_column':'closet_cost_info', # Asset value in 100 million RMB
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['lost_income_48RMB_month'], # Annual Economic loss RMB
                            'max_economic_loss_column':['lost_income_48RMB_month'], # Annual Economic loss RMB
                            'economic_loss_conversion': 1.0/365.0, # Convert Annual Losses to Daily in RMB 
                            'length_column':None,
                            'length_unit':None, # To convert length in km to meters
                            'shapefile':'landfill_nodes.shp',
                            'attribute_info':'landfill.csv',
                            'layer':'nodes',
                            'damage_cost_unit':'RMB',
                            'economic_loss_unit':'RMB/day'
                        },
                        {
                            'sector':'air',
                            'sector_type':'nodes',
                            'sector_name':'Airport',
                            'id_column':'ID',
                            'adaptation_criteria_column':'Grade',
                            'flood_protection_column':'Flood_protection ',
                            'cost_column':'best_nearest_cost', # Asset value in 100 million RMB
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['income_loss_yuan'], # Annual Economic loss RMB
                            'max_economic_loss_column':['income_loss_yuan'], # Annual Economic loss RMB
                            'economic_loss_conversion': 1.0/365.0, # Convert Annual Losses to Daily in RMB
                            'length_column':None,
                            'length_unit':None, # To convert length in km to meters
                            'shapefile':'air_nodes.shp',
                            'attribute_info':'airport.csv',
                            'layer':'nodes',
                            'damage_cost_unit':'RMB',
                            'economic_loss_unit':'RMB/day'
                        },
                        {
                            'sector':'power',
                            'sector_type':'nodes',
                            'sector_name':'Power Plant',
                            'id_column':'Number',
                            'adaptation_criteria_column':'Capacity_M',
                            'flood_protection_column':'Flood_prot',
                            'cost_column':'best_nearest_cost', # Asset value in 100 million RMB
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['cus_loss_income_total_rmb','total_loss_business16_RMB'], # Annual Economic loss RMB
                            'max_economic_loss_column':['cus_loss_income_total_rmb','total_loss_business16_RMB'], # Annual Economic loss RMB
                            'economic_loss_conversion': 1.0/365.0, # Convert Annual Losses to Daily in RMB
                            'length_column':None,
                            'length_unit':None, # To convert length in km to meters
                            'shapefile':'power_nodes.shp',
                            'attribute_info':'power plants.csv',
                            'layer':'nodes',
                            'damage_cost_unit':'RMB',
                            'economic_loss_unit':'RMB/day'
                        },
                        {
                            'sector':'road',
                            'sector_type':'edges',
                            'sector_name':'Road',
                            'id_column':'road_ID',
                            'adaptation_criteria_column':'grade',
                            'flood_protection_column':'flood_pro',
                            'cost_column':'best_cost_per_km_sec', # Asset value in 100 million RMB/km
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['loss_income_min'], # Daily Economic loss RMB
                            'max_economic_loss_column':['loss_income_max'], # Daily Economic loss RMB
                            'economic_loss_conversion': 1.0, # Convert Annual Losses to Daily in RMB
                            'length_column':'road_length_km',
                            'length_unit':1000.0, # To convert length in km to meters
                            'shapefile':'final_road.shp',
                            'attribute_info':'road.csv',
                            'layer':'edges',
                            'damage_cost_unit':'RMB/m',
                            'economic_loss_unit':'RMB/day'
                        }
                        ]

    boundary_gpd = gpd.read_file(os.path.join(incoming_data_path,
                                'province_shapefile',
                                'China_pro_pop_electricity.shp'),encoding="utf-8")
    boundary_gpd = boundary_gpd.to_crs(epsg=china_proj)
    dzm_names = [('810000',"Hong Kong"),('710000',"Taiwan"),('820000',"Macao")]
    for ix,(dzm,name) in enumerate(dzm_names):
        boundary_gpd.loc[boundary_gpd["DZM"] == dzm,"Region"] = name

    boundary_gpd = boundary_gpd[['DZM','Region','geometry']]
    boundary_gpd.to_file(os.path.join(processed_data_path,
                                                    'admin_boundaries',
                                                    'China_regions.gpkg'),driver="GPKG")

    for sector in sector_descriptions:
        sector_shape = gpd.read_file(os.path.join(incoming_data_path,
                        "assets",sector["shapefile"]))[[sector["id_column"],sector["adaptation_criteria_column"],"geometry"]]
        sector_shape = sector_shape.to_crs(epsg=china_proj)
        sector_shape = match_assets_to_boundaries(sector_shape,boundary_gpd,sector["id_column"],
                                                "Region",asset_type=sector["sector_type"])
        sector_attributes = pd.read_csv(os.path.join(incoming_data_path,"asset_attributes",sector["attribute_info"]))
        sector_attributes.rename(columns={sector["flood_protection_column"]:"design_protection_rp"},inplace=True)
        sector_attributes['asset_type'] = sector['sector']
        sector_attributes = damage_costs_and_economic_loss_estimates(sector_attributes,
        															sector)
        sector_attributes = sector_attributes[[sector["id_column"],
                                                "asset_type",
        										"design_protection_rp",
                                                "mean_damage_cost",
        										"min_damage_cost",
        										"max_damage_cost",
        										"min_economic_loss",
        										"max_economic_loss",
        										"damage_cost_unit",
        										"economic_loss_unit"
        										]]
        sector_df = pd.merge(sector_attributes,sector_shape,how="left",on=[sector['id_column']]).fillna(0)
        sector_df = gpd.GeoDataFrame(sector_df,geometry="geometry",crs=f"EPSG:{china_proj}")
        if sector['sector'] == "road":
            sector_df["road_length_km"] = sector_df.apply(lambda x: line_length_km(x.geometry),axis=1)

        sector_df.to_file(os.path.join(processed_data_path,"networks",f"{sector['sector']}.gpkg"),
                        layer=sector['layer'],driver="GPKG")




if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
