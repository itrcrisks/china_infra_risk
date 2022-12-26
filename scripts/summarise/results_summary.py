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

    climate_idx_cols = ['hazard', 'model','rcp', 'epoch','rp'] # The climate columns
    flood_protection_types = [(None,None),('Y',None)]
    flood_return_periods = [2,5,10,25,50,100,250,500,1000]

    stats_combinations = [
                            {
                            'type':'exposures',
                            'addby':[],
                            'groupby':[
                                        'rcp',
                                        'model',
                                        'epoch',
                                        'rp'
                                    ],
                            'file_name':'exposures_numbers_climate_scenario_model_year_return_period.xlsx',
                            'generate_quantiles':True 
                            },
                            {
                            'type':'exposures',
                            'addby':[],
                            'groupby':[
                                        'Region',
                                        'rcp',
                                        'model',
                                        'epoch',
                                        'rp'
                                    ],
                            'file_name':'exposures_numbers_by_region_climate_scenario_model_year_return_period.xlsx',
                            'generate_quantiles':True 
                            },
                            {
                            'type':'damages',
                            'addby':[
                                        'fragility_parameter',
                                        'damage_cost_parameter'
                                    ],
                            'groupby':[
                                        'rcp',
                                        'model',
                                        'epoch',
                                        'rp',
                                        'fragility_parameter',
                                        'damage_cost_parameter'
                                    ],
                            'file_name':'damages_numbers_climate_scenario_model_year_return_period.xlsx',
                            'generate_quantiles':True 
                            },
                            {
                            'type':'damages',
                            'addby':[
                                        'fragility_parameter',
                                        'damage_cost_parameter'
                                    ],
                            'groupby':[
                                        'Region',
                                        'rcp',
                                        'model',
                                        'epoch',
                                        'rp',
                                        'fragility_parameter',
                                        'damage_cost_parameter'
                                    ],
                            'file_name':'damages_numbers_by_region_climate_scenario_model_year_return_period.xlsx',
                            'generate_quantiles':True 
                            },
                            {
                            'type':'losses',
                            'addby':[
                                        'fragility_parameter',
                                        'economic_loss_parameter',
                                        'duration'
                                    ],
                            'groupby':[
                                        'rcp',
                                        'model',
                                        'epoch',
                                        'rp',
                                        'fragility_parameter',
                                        'economic_loss_parameter',
                                        'duration'
                                    ],
                            'file_name':'losses_numbers_climate_scenario_model_year_return_period.xlsx',
                            'generate_quantiles':True 
                            },
                            {
                            'type':'losses',
                            'addby':[
                                        'fragility_parameter',
                                        'economic_loss_parameter',
                                        'duration'
                                    ],
                            'groupby':[
                                        'Region',
                                        'rcp',
                                        'model',
                                        'epoch',
                                        'rp',
                                        'fragility_parameter',
                                        'economic_loss_parameter',
                                        'duration'
                                    ],
                            'file_name':'losses_numbers_by_region_climate_scenario_model_year_return_period.xlsx',
                            'generate_quantiles':True 
                            },
                            {
                            'type':'risks',
                            'EAD_groupby':[
                                        'rcp',
                                        'model',
                                        'epoch',
                                        'damage_cost_parameter',
                                        'fragility_parameter',
                                    ],

                            'EAEL_groupby':[
                                        'rcp',
                                        'model',
                                        'epoch',
                                        'fragility_parameter',
                                        'duration',
                                        'economic_loss_parameter',
                                    ],
                            'file_name':'risk_numbers_climate_scenario_model_year_sensitivity_parameters.xlsx',
                            'generate_quantiles':True  
                            },
                            {
                            'type':'risks_timeseries',
                            'filestring':'timeseries',
                            'EAD_groupby':[
                                        'rcp',
                                        'model',
                                        'damage_cost_parameter',
                                        'fragility_parameter',
                                    ],

                            'EAEL_groupby':[
                                        'rcp',
                                        'model',
                                        'duration',
                                        'economic_loss_parameter',
                                        'fragility_parameter',
                                        'growth_rate'
                                    ],
                            'file_name':'risk_timeseries_climate_scenario_model_year_sensitivity_parameters.xlsx',
                            'generate_quantiles':True  
                            },
                            {
                            'type':'discounted_risks_timeseries',
                            'filestring':'timeseries_discounted',
                            'EAD_groupby':[
                                        'rcp',
                                        'model',
                                        'damage_cost_parameter',
                                        'fragility_parameter',
                                        'discount_rate'
                                    ],

                            'EAEL_groupby':[
                                        'rcp',
                                        'model',
                                        'duration',
                                        'economic_loss_parameter',
                                        'fragility_parameter',
                                        'discount_rate',
                                        'growth_rate'
                                    ],
                            'file_name':'discounted_risk_timeseries_climate_scenario_model_year_sensitivity_parameters.xlsx',
                            'generate_quantiles':True  
                            },
                            {
                            'type':'risks_timeseries_assets',
                            'filestring':'timeseries',
                            'EAD_groupby':[
                                        'rcp',
                                        'model',
                                        'damage_cost_parameter',
                                        'fragility_parameter'
                                    ],

                            'EAEL_groupby':[
                                        'rcp',
                                        'model',
                                        'duration',
                                        'economic_loss_parameter',
                                        'growth_rate'
                                    ],
                            'groupby': ['rcp'],
                            'file_name':'asset_risk_timeseries_climate_scenarios_mean.xlsx',
                            'generate_quantiles':False  
                            },
                            {
                            'type':'discounted_risks_timeseries_assets',
                            'filestring':'timeseries_discounted',
                            'EAD_groupby':[
                                        'rcp',
                                        'model',
                                        'damage_cost_parameter',
                                        'fragility_parameter',
                                        'discount_rate'
                                    ],

                            'EAEL_groupby':[
                                        'rcp',
                                        'model',
                                        'duration',
                                        'economic_loss_parameter',
                                        'discount_rate',
                                        'growth_rate'
                                    ],
                            'groupby': ['rcp'],
                            'file_name':'asset_discounted_risk_timeseries_climate_scenarios_mean.xlsx',
                            'generate_quantiles':False  
                            },
                            {
                            'type':'investments',
                            'groupby':[],
                            'file_name':'investment_numbers.xlsx',
                            'generate_quantiles':False 
                            },
                            {
                            'type':'investment_timeseries',
                            'EAD_groupby':[
                                        'rcp',
                                        'model',
                                        'damage_cost_parameter',
                                        'fragility_parameter',
                                        'discount_rate'
                                    ],
                            'EAEL_groupby':[
                                        'rcp',
                                        'model',
                                        'duration',
                                        'fragility_parameter',
                                        'economic_loss_parameter',
                                        'discount_rate',
                                        'growth_rate'
                                    ],
                            'cost_groupby':[
                                        'discount_rate'
                                    ], 
                            'file_name':'investment_timeseries_climate_scenario_model_year_sensitivity_parameters.xlsx',
                            'generate_quantiles':True  
                            },
                        ]

    quantile_combinations = [
                            {
                            'type':'exposures',
                            'groupby':[
                                        'rcp',
                                        'epoch',
                                        'rp'
                                    ],
                            'file_name':'exposures_numbers_climate_scenario_year_return_period.xlsx',
                            },
                            {
                            'type':'exposures',
                            'groupby':[
                                        'Region',
                                        'rcp',
                                        'epoch',
                                        'rp'
                                    ],
                            'file_name':'exposures_numbers_by_region_climate_scenario_year_return_period.xlsx',
                            },
                            {
                            'type':'damages',
                            'groupby':[
                                        'rcp',
                                        'epoch',
                                        'rp'
                                    ],
                            'file_name':'damages_numbers_climate_scenario_year_return_period.xlsx',
                            },
                            {
                            'type':'damages',
                            'groupby':[
                                        'Region',
                                        'rcp',
                                        'epoch',
                                        'rp'
                                    ],
                            'file_name':'damages_numbers_by_region_climate_scenario_year_return_period.xlsx',
                            },
                            {
                            'type':'losses',
                            'groupby':[
                                        'rcp',
                                        'epoch',
                                        'rp'
                                    ],
                            'file_name':'losses_numbers_climate_scenario_year_return_period.xlsx',
                            },
                            {
                            'type':'losses',
                            'groupby':[
                                        'Region',
                                        'rcp',
                                        'epoch',
                                        'rp'
                                    ],
                            'file_name':'losses_numbers_by_region_climate_scenario_year_return_period.xlsx',
                            },
                            {
                            'type':'risks',
                            'groupby':[
                                        'rcp',
                                        'epoch'
                                    ],
                            'file_name':'risk_numbers_climate_scenario_year.xlsx',
                            },
                            {
                            'type':'risks_timeseries',
                            'groupby':[
                                        'rcp',
                                        'epoch'
                                    ],
                            'file_name':'risk_timeseries_climate_scenario_year.xlsx',
                            },
                            {
                            'type':'discounted_risks_timeseries',
                            'groupby':[
                                        'rcp',
                                        'epoch'
                                    ],
                            'file_name':'discounted_risk_timeseries_climate_scenario_year.xlsx',
                            },
                            {
                            'type':'risks_timeseries_assets',
                            'groupby':[
                                    ],
                            'file_name':'asset_risk_timeseries_climate_scenario_year.xlsx',
                            },
                            {
                            'type':'discounted_risks_timeseries_assets',
                            'groupby':[
                                    ],
                            'file_name':'asset_risk_timeseries_climate_scenario_year.xlsx',
                            },
                            {
                            'type':'investments',
                            'groupby':[
                                    ],
                            'file_name':None,
                            },
                            {
                            'type':'investment_timeseries',
                            'groupby':[
                                        'rcp'
                                    ],
                            'file_name':'investment_timeseries_climate_scenarios.xlsx',
                            }  
                        ]
    start_year = 2020
    end_year = 2080
    # length_division = 1000.0 # Convert m to km
    network_csv = os.path.join(processed_data_path,
                            "network_layers_hazard_intersections_details.csv")
    sector_descriptions = pd.read_csv(network_csv)

    hazard_data_details = pd.read_csv(os.path.join(processed_data_path,
                                    "hazard_layers.csv"),encoding="latin1")
    hazard_data_details = hazard_data_details.assign(key = hazard_data_details.key + '_mod')
    hazard_keys = hazard_data_details["key"].values.tolist()

    sector_asset_stats = []
    sector_asset_protect = []
    # asset_flood_protection_column = ['Flood_prot','Flood prot','Flood_prot','flood_pro']
    for asset_info in sector_descriptions.itertuples():
        asset_sector = asset_info.asset_gpkg
        asset_id = asset_info.asset_id_column
        flood_protection_column = asset_info.flood_protection_column
        asset_max_cost = asset_info.asset_max_cost_column
        asset_max_econ_loss = asset_info.asset_max_economic_loss_column

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
        for asset_info in sector_descriptions.itertuples():
            asset_sector = asset_info.asset_gpkg
            asset_id = asset_info.asset_id_column
            flood_protection_column = asset_info.flood_protection_column
            
            assets = gpd.read_file(os.path.join(processed_data_path,
                                        "networks",
                                        f"{asset_sector}.gpkg"),
                                        layer=asset_info.asset_layer)

            if stats_combinations[st]['type'] != 'investment_timeseries':
                output_excel = os.path.join(stats_path,
                                    f"{asset_sector}_{stats_combinations[st]['file_name']}")
                stats_wrtr = pd.ExcelWriter(output_excel)

            if stats_combinations[st]['generate_quantiles'] is True:
                output_excel = os.path.join(stats_path,
                                f"{asset_sector}_{quantile_combinations[st]['file_name']}")
                quantile_wrtr = pd.ExcelWriter(output_excel)

            if stats_combinations[st]['type'] in ('exposures','damages','losses'):
                for defense in ['no_protection_rp',flood_protection_column]:
                    exposures = pd.read_parquet(os.path.join(risk_results_path,asset_sector,
                                        f"{asset_sector}_{asset_info.asset_layer}_{defense}_{stats_combinations[st]['type']}.parquet")) 
                    exposures = pd.merge(exposures,assets[[asset_id,"Region"]],how="left",on=[asset_id])
                    exposures = add_rows_and_transpose(exposures,hazard_data_details,
                                        [asset_id,"Region"] + stats_combinations[st]['addby'],
                                        hazard_keys,
                                        climate_idx_cols)
                    group_and_write_results(exposures,
                        stats_combinations[st]['groupby'],stats_wrtr,
                        quantile_combinations[st]['groupby'],quantile_wrtr,
                        ["totals"],f"{asset_sector}-{defense}",
                        write_values=write_values,
                        generate_groups = True,
                        generate_quantiles=stats_combinations[st]['generate_quantiles'])
            
            elif stats_combinations[st]['type'] == 'risks':
                # defense = ['undefended','designed_protection','avoided']
                root_dir = os.path.join(risk_results_path,asset_sector)
                for root, dirs, files in os.walk(root_dir):
                    for file in files:
                        get_risk_results(root_dir,file,asset_sector,asset_id,
                                    ["hazard","model","rcp","epoch"],
                                    stats_combinations[st],stats_wrtr,
                                    quantile_combinations[st],quantile_wrtr,write_values=write_values)

            elif stats_combinations[st]['type'] == 'investment_timeseries':
                # defense = ['undefended','designed_protection','avoided']
                asset_risks = []
                root_dir = os.path.join(risk_results_path,asset_sector)
                for root, dirs, files in os.walk(root_dir):
                    for file in files:
                        asset_risks = get_cost_benefit_results(asset_risks,root_dir,
                                            file,asset_sector,asset_id,
                                            stats_combinations[st],
                                            quantile_combinations[st])

                asset_risks.to_excel(quantile_wrtr,sheet_name=asset_sector, index=False)
                # quantile_wrtr.save()

            elif stats_combinations[st]['type'] in ('risks_timeseries','discounted_risks_timeseries'):
                root_dir = os.path.join(risk_results_path,asset_sector)
                for root, dirs, files in os.walk(root_dir):
                    for file in files:
                        get_risk_timeseries_results(root_dir,file,stats_combinations[st]['filestring'],
                        			asset_sector,asset_id,
                        			start_year,end_year,
                                    stats_combinations[st],stats_wrtr,
                                    quantile_combinations[st],quantile_wrtr,write_values=write_values)
                
            elif stats_combinations[st]['type'] in ('risks_timeseries_assets','discounted_risks_timeseries_assets'):
                root_dir = os.path.join(risk_results_path,asset_sector)
                for root, dirs, files in os.walk(root_dir):
                    for file in files:
                        get_risk_timeseries_assets(root_dir,file,stats_combinations[st]['filestring'],
                        			asset_sector,asset_id,
                        			start_year,end_year,
                                    stats_combinations[st],stats_wrtr,
                                    None,None,write_values=write_values)
            elif (stats_combinations[st]['type'] == 'investments'):
                exposures = pd.read_parquet(os.path.join(risk_results_path,
                                        asset_sector,
                                        f"{asset_sector}_{asset_info.asset_layer}_design_protection_rp_npvs.parquet"
                                        ))
                exposures = exposures.drop_duplicates(subset=[asset_id],keep='first')
                group = ["mean_ini_adapt_cost_design_protection_rp"]
                for cnt in (0,2):        
                    if cnt == 0:
                       stats_combinations[st]['groupby'] = [asset_id,flood_protection_column]
                       defense = 'by-asset-protection'
                    else:
                       stats_combinations[st]['groupby'] = [flood_protection_column]  
                       defense = 'by-protection'

                    group_and_write_results(exposures,
                            stats_combinations[st]['groupby'],stats_wrtr,
                            '','',
                            group,f"{asset_sector}-{defense}",
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
