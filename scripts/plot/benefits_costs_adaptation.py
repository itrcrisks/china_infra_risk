"""Road network risks and adaptation maps
"""
import os
import sys
from collections import OrderedDict
import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
# from map_plotting_utils import (load_config,get_projection,
#                             plot_basemap, plot_point_assets, 
#                             plot_line_assets, save_fig)
from map_plotting_utils import (load_config, get_projection, 
                        plot_basemap, plot_point_assets,
                        point_map_plotting_color_width,plot_line_assets,
                        line_map_plotting_colors_width,save_fig,
                        line_map_plotting,point_map_plotting)
from tqdm import tqdm
tqdm.pandas()

def grouping_values(dataframe,grouping_by_columns,grouped_columns,line_assets=False):
    if line_assets is False:
        df = dataframe.groupby(grouping_by_columns
                            )[grouped_columns].agg(lambda x:len(x.unique())).reset_index(name='assets')

    else:
        df = dataframe.groupby(grouping_by_columns
                            )[grouped_columns].sum().reset_index()

    return df

def quantiles(dataframe,grouping_by_columns,grouped_columns):
    quantiles_list = ['mean','min','max','mean','q5','q95']
    df_list = []
    for quant in quantiles_list:
        if quant == 'mean':
            # print (dataframe)
            df = dataframe.groupby(grouping_by_columns)[grouped_columns].mean()
        elif quant == 'min':
            df = dataframe.groupby(grouping_by_columns)[grouped_columns].min()
        elif quant == 'max':
            df = dataframe.groupby(grouping_by_columns)[grouped_columns].max()
        elif quant == 'mean':
            df = dataframe.groupby(grouping_by_columns)[grouped_columns].quantile(0.5)
        elif quant == 'q5':
            df = dataframe.groupby(grouping_by_columns)[grouped_columns].quantile(0.05)
        elif quant == 'q95':
            df = dataframe.groupby(grouping_by_columns)[grouped_columns].quantile(0.95)

        df.rename(columns=dict((g,'{}_{}'.format(g,quant)) for g in grouped_columns),inplace=True)
        df_list.append(df)
    return pd.concat(df_list,axis=1).reset_index()

def change_max_depth(x):
    if isinstance(x,str):
        if x == '999m':
            return '10m'
        else:
            return x
    else:
        return x

def change_depth_string_to_number(x):
    if isinstance(x,str):
        if 'cm' in x:
            return 0.01*float(x.split('cm')[0])
        elif 'm' in x:
            return 1.0*float(x.split('m')[0])
        else:
            return float(x)
    else:
        return x

def get_protection_benefits_costs(df,protection_columns,
                            risk_cost_type,
                            benefit_column_type):
    cba = []
    for pc in protection_columns:
        fl_rp = df[pc].max()
        if risk_cost_type == "Residual Risks":
            ead_col = f'EAD_river_{pc}_npv_{benefit_column_type}'
            eael_col = f'EAEL_river_{pc}_npv_{benefit_column_type}'
            val = df[ead_col].sum() + df[eael_col].sum()
        elif risk_cost_type == "Avoided Risks":
            ead_col = f'Avoided_EAD_river_{pc}_npv_{benefit_column_type}'
            eael_col = f'Avoided_EAEL_river_{pc}_npv_{benefit_column_type}'
            val = df[ead_col].sum() + df[eael_col].sum()
        elif risk_cost_type == "Adaptation Costs":
            ini_cost_col = f'mean_ini_adapt_cost_{pc}_{pc}'
            total_cost_col = f'mean_total_adapt_cost_npv_{pc}_{benefit_column_type}'
            val = df[total_cost_col].sum()

        cba.append((fl_rp,val))

    cba = [cba[0]] + list(sorted(list(set(cba[1:])),key = lambda v:v[0], reverse=False))
    return cba

def main(config):
    data_path = config['paths']['data']
    output_path = config['paths']['output']
    figure_path = config['paths']['figures']
    network_path = os.path.join(data_path,'networks') # Where we have all the network shapefiles
    risk_results_path = os.path.join(output_path,'risk_and_adaptation_results') # Where we have all the risk results

    figures = os.path.join(figure_path,'benefits_costs')
    if os.path.exists(figures) == False:
        os.mkdir(figures)

    protection_colors = ['#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
    sector_descriptions = [
                        {
                            'sector':'landfill',
                            'sector_name':'Landfill',
                            'sector_label':'Landfill sites',
                            'asset_type':'nodes',
                            'sector_shapefile':'landfill.gpkg',
                            'id_column':'Plant_Numb',
                            'adaptation_criteria_column':'Capacity_10,000 m3',
                            'flood_protection_column':'design_protection_rp',
                            'flood_protection_column_assets':'design_protection_rp',
                            'cost_column':'closet_cost_info', # Asset value in 100 million RMB
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['lost_income_48RMB_month'], # Annual Economic loss RMB
                            'max_economic_loss_column':['lost_income_48RMB_month'], # Annual Economic loss RMB
                            'economic_loss_conversion': 1.0/365.0, # Convert Annual Losses to Daily in RMB 
                            'length_column':None,
                            'length_unit':1, # To convert length in km to meters
                            'sector_marker':'o',
                            'sector_size':12.0,
                            'sector_color':'#74c476',
                            'exposure_column':'totals'
                        },
                        {
                            'sector':'air',
                            'sector_name':'Airport',
                            'sector_label':'Airports',
                            'asset_type':'nodes',
                            'sector_shapefile':'air.gpkg',
                            'id_column':'ID',
                            'adaptation_criteria_column':'Grade',
                            'flood_protection_column':'design_protection_rp',
                            'flood_protection_column_assets':'design_protection_rp',
                            'cost_column':'best_nearest_cost', # Asset value in 100 million RMB
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['income_loss_yuan'], # Annual Economic loss RMB
                            'max_economic_loss_column':['income_loss_yuan'], # Annual Economic loss RMB
                            'economic_loss_conversion': 1.0/365.0, # Convert Annual Losses to Daily in RMB
                            'length_column':None,
                            'length_unit':1, # To convert length in km to meters
                            'sector_marker':'s',
                            'sector_size':12.0,
                            'sector_color':'#636363',
                            'exposure_column':'totals'
                        },
                        {
                            'sector':'power',
                            'sector_name':'Power Plant',
                            'sector_label':'Power plants',
                            'asset_type':'nodes',
                            'sector_shapefile':'power.gpkg',
                            'id_column':'Number',
                            'adaptation_criteria_column':'Capacity_M',
                            'flood_protection_column':'design_protection_rp',
                            'flood_protection_column_assets':'design_protection_rp',
                            'cost_column':'best_nearest_cost', # Asset value in 100 million RMB
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['cus_loss_income_total_rmb','total_loss_business16_RMB'], # Annual Economic loss RMB
                            'max_economic_loss_column':['cus_loss_income_total_rmb','total_loss_business16_RMB'], # Annual Economic loss RMB
                            'economic_loss_conversion': 1.0/365.0, # Convert Annual Losses to Daily in RMB
                            'length_column':None,
                            'length_unit':1, # To convert length in km to meters
                            'sector_marker':'^',
                            'sector_size':12.0,
                            'sector_color':'#fb6a4a',
                            'exposure_column':'totals'
                        },
                        {
                            'sector':'road',
                            'sector_name':'Road',
                            'sector_label':'Roads',
                            'asset_type':'edges',
                            'sector_shapefile':'road.gpkg',
                            'id_column':'road_ID',
                            'adaptation_criteria_column':'grade',
                            'flood_protection_column':'design_protection_rp',
                            'flood_protection_column_assets':'design_protection_rp',
                            'cost_column':'best_cost_per_km_sec', # Asset value in 100 million RMB/km
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['loss_income_min'], # Daily Economic loss RMB
                            'max_economic_loss_column':['loss_income_max'], # Daily Economic loss RMB
                            'economic_loss_conversion': 1.0, # Convert Annual Losses to Daily in RMB
                            'length_column':'road_length_km',
                            'length_unit':0.001, # To convert length in km to meters
                            'sector_marker':None,
                            'sector_size':1.0,
                            'sector_color':'#969696',
                            'exposure_column':'totals'
                        }
                        ]

    plot_types = [
                        # {
                        # 'type':'benefits_costs',
                        # 'groupby':[
                        #             'rcp'
                        #         ],
                        # 'climate_scenarios':['4.5','8.5'],
                        # 'scenario_color':['#d94801','#7f2704'], 
                        # 'scenario_marker':['s-','^-'], 
                        # 'file_name':'benefits_costs_climate_scenarios.xlsx',
                        # 'plot_name':'benefits_costs_protection_standards.png'
                        # },
                        # {
                        # 'type':'benefits_costs_modified',
                        # 'groupby':[
                        #             'rcp'
                        #         ],       
                        # 'climate_scenarios':['4.5','8.5'],
                        # 'scenario_color':['#d94801','#7f2704'], 
                        # 'scenario_marker':['s-','^-'], 
                        # 'file_name':'benefits_costs_climate_scenarios.xlsx',
                        # 'plot_name':'benefits_costs_modified_protection_standards.png'
                        # },
                        {
                        'type':'asset_optimals',
                        'groupby':[
                                    'rcp'
                                ],       
                        'climate_scenarios':['4.5','8.5'],
                        'scenario_color':['#d94801','#7f2704'], 
                        'scenario_marker':['s-','^-'], 
                        'file_name':'benefits_costs_climate_scenarios_optimals_asset_level_amax.csv',
                        'plot_name':'asset_optimal_protection_standards.png'
                        },
                    ]

    baseyear = 1980
    cost_unit = 1e-9 # To convert damage and loss estimates to RMB billions
    map_return_periods = [100.0,1000.0]
    flood_color = '#3182bd'
    noflood_color = '#969696' 
    # flood_colors = ['#c6dbef','#6baed6','#2171b5','#08306b']
    # change_colors = ['#c6dbef','#9ecae1','#6baed6','#3182bd','#08519c','#969696']
    # change_colors = ['#d73027','#fc8d59','#fee08b','#91cf60','#1a9850','#969696']
    # change_labels = ['0% - 20%','20% - 40%','40% - 60%','60% - 80%','80% - 100%','No change/value']
    #ffeda0
    #fed976
    #feb24c
    #fd8d3c
    #fc4e2a
    #e31a1c
    #bd0026
    #800026
    protection_standards_colors = [(0,'#feb24c'),
                                (10,'#fd8d3c'),
                                (20,'#fc4e2a'),
                                (25,'#e31a1c'),
                                (50,'#bd0026'),
                                (100,'#800026'),
                                ('Not flooded',noflood_color)]

    """Get all the exposure plots
    """
    protection_levels = [
                        "to_100_year_protection",
                        "to_250_year_protection",
                        "to_500_year_protection",
                        "to_1000_year_protection"
                        ]
    benefits_costs_types = ['Residual Risks','Avoided Risks','Adaptation Costs']
    flood_protection_column = "design_protection_rp"

    """
    Step 1: Get all the Administrative boundary files
        Combine the English provinces names with the Chinese ones  
    """
    boundary_gpd = gpd.read_file(os.path.join(data_path,
                                'admin_boundaries',
                                'China_regions.gpkg'),encoding="utf-8")
    # admin_gpd = gpd.read_file(os.path.join(data_path,
    #                             'Household_shapefile',
    #                             'County_pop_2018.shp'),encoding="utf-8")
    # boundary_names = pd.merge(admin_gpd[['AdminCode','ProAdCode']],
    #                 boundary_gpd[['DZM','Region']],
    #                 how='left',
    #                 left_on=['ProAdCode'],
    #                 right_on=['DZM'])
    # boundary_names['AdminCode']=boundary_names['AdminCode'].astype(int)


    """
    Step 2: Get map bounds to set the map projections  
    """
    bounds = boundary_gpd.geometry.total_bounds # this gives your boundaries of the map as (xmin,ymin,xmax,ymax)
    ax_proj = get_projection(extent = (bounds[0]+5,bounds[2]-10,bounds[1],bounds[3]))
    del boundary_gpd

    for st in range(len(plot_types)):    
        if plot_types[st]["type"] != "asset_optimals":
            figure_texts = ['a.','b.','c.','d.','e.','f.','g.','h.','i.','j.','k.','l.']
            quantiles_list = ['mean','amin','amax']
            fig, ax_plots = plt.subplots(3,4,
                    figsize=(30,16),
                    dpi=500)
            ax_plots = ax_plots.flatten()
            j = 0
            for bct in benefits_costs_types:
                for s in range(len(sector_descriptions)):
                    sector = sector_descriptions[s]
                    benefits_costs = pd.read_excel(os.path.join(output_path,
                                            plot_types[st]['type'],
                                            f"{sector['sector']}_{plot_types[st]['file_name']}"),
                                            sheet_name=f"{sector['sector']}")
                    benefits_costs = benefits_costs[benefits_costs["EAD_river_no_protection_rp_npv_amax"]>0]
                    pr_cols = list(set([col for col in benefits_costs.columns.values.tolist() if '_to_' in col]))
                    protection_columns = []
                    for col in pr_cols: 
                        fn = col.split('_to_')
                        protection_columns.append(f"{fn[0].split('_')[-1]}_to_{fn[1].split('_')[0]}_year_protection")

                    protection_columns = list(set(protection_columns))
                    protection_select = []
                    for pr in protection_levels:
                        protection_select += [p for p in protection_columns if pr in p]

                    protection_columns = [flood_protection_column] + list(set(protection_select))
                    
                    ax = ax_plots[j]
                    for c in range(len(plot_types[st]['climate_scenarios'])):
                        sc = float(plot_types[st]['climate_scenarios'][c])
                        cl = plot_types[st]['scenario_color'][c]
                        m = plot_types[st]['scenario_marker'][c]
                        mean_bc_vals = get_protection_benefits_costs(benefits_costs[benefits_costs["rcp"] == sc],
                                                            protection_columns,
                                                            bct,
                                                            'mean')
                        rps, mean_vals = zip(*mean_bc_vals)

                        min_bc_vals = get_protection_benefits_costs(benefits_costs[benefits_costs["rcp"] == sc],
                                                            protection_columns,
                                                            bct,
                                                            'amin')
                        rps, min_vals = zip(*min_bc_vals)

                        max_bc_vals = get_protection_benefits_costs(benefits_costs[benefits_costs["rcp"] == sc],
                                                            protection_columns,
                                                            bct,
                                                            'amax')
                        rps, max_vals = zip(*max_bc_vals)
                        x_vals = 1 + np.arange(len(rps))
                        x_labels = ["Design RP"] + [f"{y}-year RP" for y in rps[1:]]
                        ax.plot(np.array(x_vals),
                                cost_unit*np.array(mean_vals),
                                m,color=cl,markersize=10,linewidth=2.0,
                                label=f"RCP {str(sc).upper()} mean")
                        ax.fill_between(np.array(x_vals),cost_unit*np.array(min_vals),
                            cost_unit*np.array(max_vals),alpha=0.3,facecolor=cl,
                            label=f"RCP {str(sc).upper()} min-max")
                    
                        ax.set_xlabel('Design Levels',fontsize=14,fontweight='bold')
                        ax.set_ylabel(f"{bct} PV (RMB billion)",fontsize=14,fontweight='bold')
                        # if bct == "Avoided Risks":
                        #     ax.set_yscale('log')

                        ax.set_xticks(x_vals)
                        ax.set_xticklabels(x_labels)
                        ax.grid(True)
                        ax.text(
                            0.05,
                            0.95,
                            figure_texts[j],
                            horizontalalignment='left',
                            transform=ax.transAxes,
                            size=18,
                            weight='bold')
                        ax.text(
                            0.15,
                            0.95,
                            f"{bct} PV",
                            horizontalalignment='left',
                            transform=ax.transAxes,
                            size=18,
                            weight='bold')
                        if j <= 3:
                            ax.set_title(
                                    sector['sector_label'],
                                    fontdict = {'fontsize':18,
                                    'fontweight':'bold'})

                    j+=1            

            ax_plots[7].legend(
                        loc='upper left', 
                        bbox_to_anchor=(1.05,0.8),
                        prop={'size':18,'weight':'bold'})
            plt.tight_layout()
            save_fig(os.path.join(figures,plot_types[st]['plot_name']))
            plt.close()
        elif plot_types[st]["type"] == "asset_optimals":
            """Get all the exposure plots
            """
            investment_column = 'optimal_flood_protection_amax_total_mean_fit'
            figure_texts = ['a.','b.','c.','d.','e.','f.','g.','h.']
            fig, ax_plots = plt.subplots(2,4,
                    subplot_kw={'projection': ax_proj},
                    figsize=(24,10),
                    dpi=500)
            ax_plots = ax_plots.flatten()
            j = 0
            for sc in [4.5,8.5]:
                for s in range(len(sector_descriptions)):
                    legend_handles = []
                    sector = sector_descriptions[s]
                    assets = gpd.read_file(os.path.join(network_path,
                                                sector['sector_shapefile']),layer=sector['asset_type'])
                    investments = pd.read_csv(os.path.join(output_path,
                                    'benefits_costs_modified',
                                    f"{sector['sector']}_{plot_types[st]['file_name']}"))[[
                                    sector['id_column'],'rcp',investment_column]]
                    investments = investments[investments["rcp"] == sc]
                    flood_ids = list(set(investments[sector['id_column']].values.tolist()))
                    assets = gpd.GeoDataFrame(pd.merge(assets,
                                            investments[[sector['id_column'],investment_column]],
                                            how='left',on=[sector['id_column']]),
                                            geometry='geometry',crs='epsg:4326')

                    assets[investment_column].fillna(0,inplace=True)
                    # assets[investment_column] = assets[investment_column]
                    # assets[investment_column] = assets.apply(lambda x: modify_investment(x,sector['flood_protection_column_assets'],investment_column),axis=1) 
                    ax = plot_basemap(ax_plots[j],include_labels=False)
                    if sector['asset_type'] == 'nodes':
                        ax = point_map_plotting_color_width(ax,assets,investment_column,
                                sector['sector_marker'],1.0,
                                "Optimal\ Standard (years)",'flooding',
                                point_colors = protection_colors,
                                no_value_color = '#969696',
                                point_steps = len(protection_colors),
                                interpolation='fisher-jenks'
                                )
                    else:
                        # assets[investment_column] = sector['length_unit']*assets[investment_column]
                        # assets[[sector['id_column'],sector['flood_protection_column'],investment_column]].to_csv('text.csv',index=False)
                        ax = line_map_plotting_colors_width(ax,assets,investment_column,
                                    1.0,
                                    'Optimal Standard (years)',
                                    'flooding',
                                    line_colors = protection_colors,
                                    no_value_color = '#969696',
                                    line_steps = len(protection_colors),
                                    width_step = 0.02,
                                    significance=0)
                    ax.text(
                        0.05,
                        0.95,
                        figure_texts[j],
                        horizontalalignment='left',
                        transform=ax.transAxes,
                        size=18,
                        weight='bold')
                    ax.set_title(
                                sector['sector_label'],
                                fontdict = {'fontsize':18,
                                'fontweight':'bold'})


                    # ax.legend(handles=legend_handles,fontsize=10,title="$\\bf{Protection\ standard\ (years)}$",loc='lower left')
                    ax.text(
                        0.15,
                        0.95,
                        f"RCP {sc}",
                        horizontalalignment='left',
                        transform=ax.transAxes,
                        size=18,
                        weight='bold')     

                    j+=1            

            plt.tight_layout()
            save_fig(os.path.join(figures,'optimal_flood_protection_return_periods.png'))
            plt.close()


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
