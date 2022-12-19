"""Road network risks and adaptation maps
"""
import os
import sys
from collections import OrderedDict
import pandas as pd
import geopandas as gpd
import zipfile
import numpy as np
import ast
from summary_utils import *
from tqdm import tqdm
tqdm.pandas()


def main(config):
    processed_data_path = config['paths']['data']
    output_path = config['paths']['output']

    # network_path = os.path.join(data_path,'network') # Where we have all the network shapefiles
    # asset_details_path = os.path.join(data_path,'asset_details') # Where we have all the exposure results
    risk_results_path = os.path.join(output_path,'risk_and_adaptation_results') # Where we have all the risk results

    stats_path = os.path.join(output_path,'exposures')
    if os.path.exists(stats_path) == False:
        os.mkdir(stats_path)

    sector_descriptions = [
                        {
                            'results_folder_type': 'Normal', # Folder type is either Zip or Normal
                            'sector':'landfill',
                            'sector_name':'Landfill',
                            'sector_label':'Landfill sites',
                            'asset_type':'nodes',
                            'sector_shapefile':'landfill_nodes.shp',
                            'id_column':'Plant_Numb',
                            'adaptation_criteria_column':'Capacity_10,000 m3',
                            'flood_protection_column':'Flood_protect',
                            'flood_protection_column_assets':'Flood_prot',
                            'cost_column':'closet_cost_info', # Asset value in 100 million RMB
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['lost_income_48RMB_month'], # Annual Economic loss RMB
                            'max_economic_loss_column':['lost_income_48RMB_month'], # Annual Economic loss RMB
                            'economic_loss_conversion': 1.0/365.0, # Convert Annual Losses to Daily in RMB 
                            'length_column':None,
                            'length_unit':None # To convert length in km to meters
                        },
                        {
                            'results_folder_type': 'Normal', # Folder type is either Zip or Normal
                            'sector':'air',
                            'sector_name':'Airport',
                            'sector_label':'Airports',
                            'asset_type':'nodes',
                            'sector_shapefile':'air_nodes.shp',
                            'id_column':'ID',
                            'adaptation_criteria_column':'Grade',
                            'flood_protection_column':'Flood_protection',
                            'flood_protection_column_assets':'Flood prot',
                            'cost_column':'best_nearest_cost', # Asset value in 100 million RMB
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['income_loss_yuan'], # Annual Economic loss RMB
                            'max_economic_loss_column':['income_loss_yuan'], # Annual Economic loss RMB
                            'economic_loss_conversion': 1.0/365.0, # Convert Annual Losses to Daily in RMB
                            'length_column':None,
                            'length_unit':None # To convert length in km to meters
                        },
                        {
                            'results_folder_type': 'Normal', # Folder type is either Zip or Normal
                            'sector':'power',
                            'sector_name':'Power Plant',
                            'sector_label':'Power plants',
                            'asset_type':'nodes',
                            'sector_shapefile':'power_nodes.shp',
                            'id_column':'Number',
                            'adaptation_criteria_column':'Capacity_M',
                            'flood_protection_column':'Flood_prot',
                            'flood_protection_column_assets':'Flood_prot',
                            'cost_column':'best_nearest_cost', # Asset value in 100 million RMB
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['cus_loss_income_total_rmb','total_loss_business16_RMB'], # Annual Economic loss RMB
                            'max_economic_loss_column':['cus_loss_income_total_rmb','total_loss_business16_RMB'], # Annual Economic loss RMB
                            'economic_loss_conversion': 1.0/365.0, # Convert Annual Losses to Daily in RMB
                            'length_column':None,
                            'length_unit':None # To convert length in km to meters
                        },
                        # {
                        #     'results_folder_type': 'Normal', # Folder type is either Zip or Normal
                        #     'sector':'road',
                        #     'sector_name':'Road',
                        #     'sector_label':'Roads',
                        #     'asset_type':'edges',
                        #     'sector_shapefile':'final_road.shp',
                        #     'id_column':'road_ID',
                        #     'adaptation_criteria_column':'grade',
                        #     'flood_protection_column':'flood_pro',
                        #     'flood_protection_column_assets':'flood_pro',
                        #     'cost_column':'best_cost_per_km_sec', # Asset value in 100 million RMB/km
                        #     'cost_conversion':1.0e8, # Convert Asset value to RMB
                        #     'min_economic_loss_column':['loss_income_min'], # Daily Economic loss RMB
                        #     'max_economic_loss_column':['loss_income_max'], # Daily Economic loss RMB
                        #     'economic_loss_conversion': 1.0, # Convert Annual Losses to Daily in RMB
                        #     'length_column':'road_length_km',
                        #     'length_unit':1000.0 # To convert length in km to meters
                        # }
                        ]

    stats_combinations = [
                            # {
                            # 'type':'exposures',
                            # 'groupby':[
                            #             'climate_scenario',
                            #             'model',
                            #             'year',
                            #             'return_period'
                            #         ],
                            # 'file_name':'exposure_numbers_climate_scenario_model_year_return_period.xlsx',
                            # 'generate_quantiles':True 
                            # },
                            # {
                            # 'type':'exposures',
                            # 'groupby':[
                            #             'Region',
                            #             'climate_scenario',
                            #             'model',
                            #             'year',
                            #             'return_period'
                            #         ],
                            # 'file_name':'exposure_numbers_by_region_climate_scenario_model_year_return_period.xlsx',
                            # 'generate_quantiles':True 
                            # },
                            # {
                            # 'type':'risks',
                            # 'EAD_groupby':[
                            #             'climate_scenario',
                            #             'model',
                            #             'year',
                            #             'flood_parameter',
                            #             'fragility_parameter',
                            #         ],

                            # 'EAEL_groupby':[
                            #             'climate_scenario',
                            #             'model',
                            #             'year',
                            #             'duration',
                            #             'economic_parameter',
                            #         ],
                            # 'file_name':'risk_numbers_climate_scenario_model_year_sensitivity_parameters.xlsx',
                            # 'generate_quantiles':True  
                            # },
                            # {
                            # 'type':'risks_timeseries',
                            # 'EAD_groupby':[
                            #             'climate_scenario',
                            #             'model',
                            #             'flood_parameter',
                            #             'fragility_parameter',
                            #             'discount_rate'
                            #         ],

                            # 'EAEL_groupby':[
                            #             'climate_scenario',
                            #             'model',
                            #             'duration',
                            #             'economic_parameter',
                            #             'discount_rate',
                            #             'gpd_growth_fluctuate'
                            #         ],
                            # 'file_name':'risk_timeseries_climate_scenario_model_year_sensitivity_parameters.xlsx',
                            # 'generate_quantiles':True  
                            # },
                            # {
                            # 'type':'risks_timeseries_assets',
                            # 'EAD_groupby':[
                            #             'climate_scenario',
                            #             'model',
                            #             'flood_parameter',
                            #             'fragility_parameter',
                            #             'discount_rate'
                            #         ],

                            # 'EAEL_groupby':[
                            #             'climate_scenario',
                            #             'model',
                            #             'duration',
                            #             'economic_parameter',
                            #             'discount_rate',
                            #             'gpd_growth_fluctuate'
                            #         ],
                            # 'groupby': ['climate_scenario'],
                            # 'file_name':'asset_risk_timeseries_climate_scenarios_median.xlsx',
                            # 'generate_quantiles':False  
                            # },
                            # {
                            # 'type':'investments',
                            # 'groupby':[],
                            # 'file_name':'investment_numbers.xlsx',
                            # 'generate_quantiles':False 
                            # },
                            {
                            'type':'investment_timeseries',
                            'EAD_groupby':[
                                        'climate_scenario',
                                        'model',
                                        'flood_parameter',
                                        'fragility_parameter',
                                        'discount_rate'
                                    ],
                            'EAEL_groupby':[
                                        'climate_scenario',
                                        'model',
                                        'duration',
                                        'economic_parameter',
                                        'discount_rate',
                                        'gpd_growth_fluctuate'
                                    ],
                            'file_name':'investment_timeseries_climate_scenario_model_year_sensitivity_parameters.xlsx',
                            'generate_quantiles':True  
                            },
                        ]

    quantile_combinations = [
                            # {
                            # 'type':'exposures',
                            # 'groupby':[
                            #             'climate_scenario',
                            #             'year',
                            #             'return_period'
                            #         ],
                            # 'file_name':'exposure_numbers_climate_scenario_year_return_period.xlsx',
                            # },
                            # {
                            # 'type':'exposures',
                            # 'groupby':[
                            #             'Region',
                            #             'climate_scenario',
                            #             'year',
                            #             'return_period'
                            #         ],
                            # 'file_name':'exposure_numbers_by_region_climate_scenario_year_return_period.xlsx',
                            # },
                            # {
                            # 'type':'risks',
                            # 'groupby':[
                            #             'climate_scenario',
                            #             'year'
                            #         ],
                            # 'file_name':'risk_numbers_climate_scenario_year.xlsx',
                            # },
                            # {
                            # 'type':'risks_timeseries',
                            # 'groupby':[
                            #             'climate_scenario',
                            #             'year'
                            #         ],
                            # 'file_name':'risk_timeseries_climate_scenario_year.xlsx',
                            # },
                            # {
                            # 'type':'risks_timeseries_assets',
                            # 'groupby':[
                            #         ],
                            # 'file_name':None,
                            # },
                            # {
                            # 'type':'investments',
                            # 'groupby':[
                            #         ],
                            # 'file_name':None,
                            # },
                            {
                            'type':'investment_timeseries',
                            'groupby':[
                                        'climate_scenario'
                                    ],
                            'file_name':'investment_timeseries_climate_scenarios.xlsx',
                            }  
                        ]
    start_year = 2020
    end_year = 2080
    # length_division = 1000.0 # Convert m to km
    """
    Step 1: Get all the Administrative boundary files
        Combine the English provinces names with the Chinese ones  
    """
    # boundary_gpd = gpd.read_file(os.path.join(data_path,
    #                             'province_shapefile',
    #                             'China_pro_pop_electricity.shp'),encoding="utf-8")
    # admin_gpd = gpd.read_file(os.path.join(data_path,
    #                             'Household_shapefile',
    #                             'County_pop_2018.shp'),encoding="utf-8")
    # boundary_names = pd.merge(admin_gpd[['AdminCode','ProAdCode']],
    #                 boundary_gpd[['DZM','Region']],
    #                 how='left',
    #                 left_on=['ProAdCode'],
    #                 right_on=['DZM'])
    # boundary_names['AdminCode']=boundary_names['AdminCode'].astype(int)

    network_csv = os.path.join(processed_data_path,
                            "network_layers_hazard_intersections_details_0.csv")
    sector_descriptions = pd.read_csv(network_csv)

    sector_asset_stats = []
    sector_asset_protect = []
    # asset_flood_protection_column = ['Flood_prot','Flood prot','Flood_prot','flood_pro']
    for asset_info in sector_descriptions.itertuples():
        asset_sector = asset_info.asset_gpkg
        asset_id = asset_info.asset_id_column
        flood_protection_column = asset_info.flood_protection_column
        # asset_min_cost = asset_info.asset_min_cost_column 
        asset_max_cost = asset_info.asset_max_cost_column
        # asset_cost_unit = asset_info.asset_cost_unit_column
        # asset_min_econ_loss = asset_info.asset_min_economic_loss_column
        asset_max_econ_loss = asset_info.asset_max_economic_loss_column
        # asset_loss_unit = asset_info.asset_economic_loss_unit_column
        # asset_type = asset_info.flooding_asset_damage_lookup_column

        assets = gpd.read_file(os.path.join(processed_data_path,
                                        "networks",
                                        f"{asset_sector}.gpkg"),
                                        layer=asset_info.asset_layer)

        assets_protection = grouping_values(assets,flood_protection_column,asset_id,add_values=False)
        assets_protection.rename(columns={flood_protection_column:'Existing protection (years)'},inplace=True)
        assets_protection['Sector'] = str(asset_info.asset_description).title()
        sector_asset_protect.append(assets_protection)
        del assets_protection

        exposures = pd.read_parquet(os.path.join(risk_results_path,asset_sector,
                        f"{asset_sector}_{asset_info.asset_layer}_{flood_protection_column}_exposures.parquet")) 
        exposures = pd.merge(exposures,assets[[asset_id,asset_max_cost,asset_max_econ_loss]],how="left",on=[asset_id])       
        sector_asset_stats.append((asset_sector,
                                len(assets.index),
                                len(exposures.index),
                                len(exposures[exposures[asset_max_cost]>0].index),
                                len(exposures[exposures[asset_max_econ_loss]>0].index)))
        if asset_sector == 'road':
            exposures = pd.merge(exposures,assets[[asset_id,"road_length_km"]],how="left",on=[asset_id])
            sector_asset_stats.append(
                                        (
                                            'Road (km)',
                                            assets['road_length_km'].sum(),
                                            exposures['road_length_km'].sum(),
                                            exposures[exposures[asset_max_cost]>0]['road_length_km'].sum(),
                                            exposures[exposures[asset_max_econ_loss]>0]['road_length_km'].sum()
                                        )    
                                    )
            assets_protection = grouping_values(assets,flood_protection_column,'road_length_km',add_values=True)
            assets_protection.rename(columns={flood_protection_column:'Existing protection (years)',
                                                'road_length_km':'assets'},inplace=True)
            assets_protection['Sector'] = 'Road (km)'
            sector_asset_protect.append(assets_protection)
            del assets_protection

    sector_asset_stats = pd.DataFrame(sector_asset_stats,
                    columns=['Sector Name',
                            'Total Assets',
                            'Total Exposed assets',
                            'Exposed assets with Asset cost > 0',
                            'Exposed assets with Economic activity > 0'
                            ])
    for c in [v for v in sector_asset_stats.columns.values.tolist() if v != "Sector Name"]:
        sector_asset_stats[c] = sector_asset_stats[c].apply(lambda v:f"{int(v):,}" if v > 999 else v)
    sector_asset_stats.to_csv(os.path.join(stats_path,'overall_exposures.csv'),index=False)
    sector_asset_protect = pd.concat(sector_asset_protect,ignore_index=True,sort='False',axis=0)[['Sector',
                                                                                                'Existing protection (years)',
                                                                                                'assets']]
    for c in [v for v in sector_asset_protect.columns.values.tolist() if v != "Sector"]:
        sector_asset_protect[c] = sector_asset_protect[c].apply(lambda v:f"{int(v):,}" if v > 999 else v)
    sector_asset_protect.to_csv(os.path.join(stats_path,'overall_protection.csv'),index=False)

    for st in range(len(stats_combinations)):
        stats_path = os.path.join(output_path,stats_combinations[st]['type'])
        if os.path.exists(stats_path) == False:
            os.mkdir(stats_path)
        # output_excel = os.path.join(stats_path,stats_combinations[st]['file_name'])
        # stats_wrtr = pd.ExcelWriter(output_excel)

        # if stats_combinations[st]['generate_quantiles'] is True:
        #     output_excel = os.path.join(stats_path,quantile_combinations[st]['file_name'])
        #     quantile_wrtr = pd.ExcelWriter(output_excel)
        write_values = True    
        for s in range(len(sector_descriptions)):
            sector = sector_descriptions[s]
            
            if stats_combinations[st]['type'] != 'investment_timeseries':
                output_excel = os.path.join(stats_path,
                                    f"{sector['sector']}_{stats_combinations[st]['file_name']}")
                stats_wrtr = pd.ExcelWriter(output_excel)

            if stats_combinations[st]['generate_quantiles'] is True:
                output_excel = os.path.join(stats_path,
                                f"{sector['sector']}_{quantile_combinations[st]['file_name']}")
                quantile_wrtr = pd.ExcelWriter(output_excel)

            assets = gpd.read_file(os.path.join(network_path,
                                sector['sector_shapefile']))
            if 'Region' in assets.columns.values.tolist():
                assets.drop(['Region'],
                                axis=1,
                                inplace=True)
            if stats_combinations[st]['type'] == 'exposures':
                for defense in ['undefended','designed_protection']:
                    exposures = pd.read_csv(os.path.join(asset_details_path,
                                            f"{sector['sector']}_intersection_cost_final.csv"
                                            ))
                    if 'probability' in exposures.columns.values.tolist():
                        exposures['return_period'] = 1/exposures['probability']
                    if defense == 'designed_protection':
                        exposures = exposures[exposures['return_period'] > exposures[sector['flood_protection_column']]]
                    if sector['asset_type'] == 'edges':
                        exposures['length'] = 1.0*exposures['length']/sector['length_unit']
                        group = ['length']
                        add_vals = True
                    else:
                        group = sector['id_column']
                        add_vals = False
                    if 'Region' in exposures.columns.values.tolist():
                        exposures.drop(['Region'],
                                        axis=1,
                                        inplace=True)
                    if 'AdminCode' in exposures.columns.values.tolist():
                        exposures['AdminCode'] = exposures['AdminCode'].astype(int)
                        exposures = pd.merge(exposures,
                                                boundary_names,
                                                how='left',
                                                on=['AdminCode'])
                    elif 'AdminCode_x' in exposures.columns.values.tolist():
                        exposures['AdminCode_x'] = exposures['AdminCode_x'].astype(int)
                        exposures = pd.merge(exposures,
                                            boundary_names,
                                            how='left',
                                            left_on=['AdminCode_x'],
                                            right_on=['AdminCode'])

                    group_and_write_results(exposures,
                        stats_combinations[st]['groupby'],stats_wrtr,
                        quantile_combinations[st]['groupby'],quantile_wrtr,
                        group,f"{sector['sector']}-{defense}",
                        add_values=add_vals,write_values=write_values,
                        generate_groups = True,
                        generate_quantiles=stats_combinations[st]['generate_quantiles'])
            elif stats_combinations[st]['type'] == 'risks':
                # defense = ['undefended','designed_protection','avoided']
                if sector['results_folder_type'] == 'Zip':
                    root_dir = os.path.join(risk_results_path,f"{sector['sector']}.zip")
                    with zipfile.ZipFile(root_dir) as zip_file:
                        for z in zip_file.infolist():
                            file = z.filename
                            get_risk_results(zip_file,file,sector,
                                        stats_combinations[st],stats_wrtr,
                                        quantile_combinations[st],quantile_wrtr,write_values=write_values)
                else:    
                    root_dir = os.path.join(risk_results_path,sector['sector'])
                    for root, dirs, files in os.walk(root_dir):
                        for file in files:
                            get_risk_results(root_dir,file,sector,
                                        stats_combinations[st],stats_wrtr,
                                        quantile_combinations[st],quantile_wrtr,write_values=write_values)

            elif stats_combinations[st]['type'] == 'investment_timeseries':
                # defense = ['undefended','designed_protection','avoided']
                asset_risks = []
                if sector['results_folder_type'] == 'Zip':
                    root_dir = os.path.join(risk_results_path,f"{sector['sector']}.zip")
                    with zipfile.ZipFile(root_dir) as zip_file:
                        for z in zip_file.infolist():
                            file = z.filename
                            asset_risks = get_cost_benefit_results(asset_risks,zip_file,
                                                file,sector,
                                                stats_combinations[st],
                                                quantile_combinations[st])
                            
                else:    
                    root_dir = os.path.join(risk_results_path,sector['sector'])
                    for root, dirs, files in os.walk(root_dir):
                        for file in files:
                            asset_risks = get_cost_benefit_results(asset_risks,root_dir,
                                                file,sector,
                                                stats_combinations[st],
                                                quantile_combinations[st])


                asset_risks.to_excel(quantile_wrtr,sheet_name=sector['sector'], index=False)
                # quantile_wrtr.save()

            elif stats_combinations[st]['type'] == 'risks_timeseries':
                if sector['results_folder_type'] == 'Zip':
                    root_dir = os.path.join(risk_results_path,f"{sector['sector']}.zip")
                    with zipfile.ZipFile(root_dir) as zip_file:
                        for z in zip_file.infolist():
                            file = z.filename
                            get_risk_timeseries_results(zip_file,file,sector,start_year,end_year,
                                        stats_combinations[st],stats_wrtr,
                                        quantile_combinations[st],quantile_wrtr,write_values=write_values)
                else:    
                    root_dir = os.path.join(risk_results_path,sector['sector'])
                    for root, dirs, files in os.walk(root_dir):
                        for file in files:
                            get_risk_timeseries_results(root_dir,file,sector,start_year,end_year,
                                        stats_combinations[st],stats_wrtr,
                                        quantile_combinations[st],quantile_wrtr,write_values=write_values)
                
            elif stats_combinations[st]['type'] == 'risks_timeseries_assets':
                if sector['results_folder_type'] == 'Zip':
                    root_dir = os.path.join(risk_results_path,f"{sector['sector']}.zip")
                    with zipfile.ZipFile(root_dir) as zip_file:
                        for z in zip_file.infolist():
                            file = z.filename
                            get_risk_timeseries_assets(zip_file,file,
                                        sector,start_year,end_year,
                                        stats_combinations[st],stats_wrtr,
                                        write_values=write_values)
                else:    
                    root_dir = os.path.join(risk_results_path,sector['sector'])
                    for root, dirs, files in os.walk(root_dir):
                        for file in files:
                            get_risk_timeseries_assets(root_dir,file,
                                        sector,start_year,end_year,
                                        stats_combinations[st],stats_wrtr,
                                        write_values=write_values)
            elif (stats_combinations[st]['type'] == 'investments'):
                exposures = pd.read_csv(os.path.join(risk_results_path,
                                        sector['sector'],
                                        f"{sector['sector']}_designed_protection_risks.csv"
                                        ))
                exposures = exposures.drop_duplicates(subset=[sector['id_column']],keep='first')
                group = [f"{iv}_ini_adapt_cost_designed_protection" for iv in ['min','max','mean','median']]
                for investment_column in group:
                    exposures[investment_column] = exposures.apply(
                                                        lambda x: modify_investment(
                                                            x,
                                                            sector['flood_protection_column'],
                                                            investment_column
                                                            ),axis=1)
                    if sector['sector'] == 'road':
                        exposures[investment_column] = sector['length_unit']*exposures[investment_column]

                for cnt in (0,2):        
                    if cnt == 0:
                       stats_combinations[st]['groupby'] = [sector['id_column'],sector['flood_protection_column']]
                       defense = 'by-asset-protection'
                    else:
                       stats_combinations[st]['groupby'] = [sector['flood_protection_column']]  
                       defense = 'by-protection'

                    group_and_write_results(exposures,
                            stats_combinations[st]['groupby'],stats_wrtr,
                            '','',
                            group,f"{sector['sector']}-{defense}",
                            add_values=True,write_values=write_values,
                            generate_groups = True,
                            generate_quantiles=False)
            if stats_combinations[st]['type'] != 'investment_timeseries':
                stats_wrtr.save()
            if stats_combinations[st]['generate_quantiles'] is True:
                quantile_wrtr.save()

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
