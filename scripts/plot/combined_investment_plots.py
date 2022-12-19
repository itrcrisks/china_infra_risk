"""Road network risks and adaptation maps
"""
import os
import sys
from collections import OrderedDict
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
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


def flood_design_protection(v,probabilites,flood_protection_column):
    prob_risk = sorted([(p,getattr(v,str(p))) for p in probabilites],key=lambda x: x[0])
    probability_threshold = getattr(v,flood_protection_column)
    if probability_threshold > 0:
        st_below = [pr for pr in prob_risk if pr[0] > 1.0/probability_threshold][0]
        st_above = [pr for pr in prob_risk if pr[0] < 1.0/probability_threshold and pr[1] > 0][-1]

        design_depth = st_below[1]  + (st_above[1] - st_below[1])*(probability_threshold - 1/st_below[0])/(1/st_above[0] - 1/st_below[0])
    else:
        design_depth = 0    

    return design_depth

def modify_investment(x,flood_protection_column,investment_column):
    if x[flood_protection_column] == 0:
        return x[investment_column]/1e3
    else:
        return x[investment_column]

def main(config):
    data_path = config['paths']['data']
    output_path = config['paths']['output']
    figure_path = config['paths']['figures']
    network_path = os.path.join(data_path,'networks') # Where we have all the network shapefiles
    asset_details_path = os.path.join(data_path,'asset_details') # Where we have all the exposure results
    risk_results_path = os.path.join(output_path,'risk_and_adaptation_results') # Where we have all the risk results


    figures = os.path.join(figure_path,'investments')
    if os.path.exists(figures) == False:
        os.mkdir(figures)

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
    baseyear = 2016
    map_return_periods = [100.0,1000.0]
    division_factor = 1e6
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
    # protection_colors = ['#feb24c','#fd8d3c','#e31a1c','#800026']
    protection_colors = ['#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
    protection_colors_roads = ['#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
    protection_standards_colors = [(10,'#feb24c'),
                                (25,'#fd8d3c'),
                                (20,'#fc4e2a'),
                                (50,'#e31a1c'),
                                # (50,'#bd0026'),
                                (100,'#800026'),
                                ('Not flooded',noflood_color)]

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


    """Get all the exposure plots
    """
    investment_column = 'mean_ini_adapt_cost_design_protection_rp'
    figure_texts = ['a.','b.','c.','d.','e.','f.','g.','h.']
    fig, ax_plots = plt.subplots(2,4,
            subplot_kw={'projection': ax_proj},
            figsize=(24,10),
            dpi=500)
    ax_plots = ax_plots.flatten()
    j = 0
    for s in range(len(sector_descriptions)):
        legend_handles = []
        sector = sector_descriptions[s]
        assets = gpd.read_file(os.path.join(network_path,
                                    sector['sector_shapefile']),layer=sector['asset_type'])
        investments = pd.read_excel(os.path.join(output_path,
                        'investments',
                        f"{sector['sector']}_investment_numbers.xlsx"),
                        sheet_name=f"{sector['sector']}-by-asset-protection")

        exposures = pd.read_parquet(os.path.join(risk_results_path,sector['sector'],
                        f"{sector['sector']}_{sector['asset_type']}_{sector['flood_protection_column']}_exposures.parquet")) 
        flood_ids = list(set(exposures[sector['id_column']].values.tolist()))
        del exposures


        assets = gpd.GeoDataFrame(pd.merge(assets,
                                investments[[sector['id_column'],investment_column]],
                                how='left',on=[sector['id_column']]),
                                geometry='geometry',crs='epsg:4326')

        assets[investment_column].fillna(0,inplace=True)
        assets[investment_column] = assets[investment_column]/division_factor
        assets[investment_column] = assets.apply(lambda x: modify_investment(x,sector['flood_protection_column_assets'],investment_column),axis=1) 
        
        ax = plot_basemap(ax_plots[j],include_labels=False)
        if sector['asset_type'] == 'nodes':
            ax = point_map_plotting_color_width(ax,assets,investment_column,
                    sector['sector_marker'],1.0,
                    'Investment (RMB million)','flooding',
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
                        'Investment (RMB million)',
                        'flooding',
                        line_colors = protection_colors_roads,
                        no_value_color = '#969696',
                        line_steps = len(protection_colors_roads),
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

        ax = plot_basemap(ax_plots[j+4],include_labels=False)
        for psc in protection_standards_colors:
            if psc[0] == 'Not flooded':
                plot_assets = assets[~assets[sector['id_column']].isin(flood_ids)]
                z_order = 7
            else:
                plot_assets = assets[(assets[sector['id_column']].isin(flood_ids)) & (assets[sector['flood_protection_column_assets']] == psc[0])]
                z_order = 8

            if len(plot_assets.index) > 0:
                if sector['asset_type'] == 'nodes':
                    ax = plot_point_assets(ax,plot_assets,
                                        psc[1],
                                        sector['sector_size'],
                                        sector['sector_marker'],
                                        z_order)
                    legend_handles.append(plt.plot([],[],
                                        marker=sector['sector_marker'], 
                                        ms=sector['sector_size'], 
                                        ls="",
                                        color=psc[1],
                                        label=psc[0])[0])
                else:
                    ax = plot_line_assets(ax,plot_assets,
                                        psc[1],
                                        sector['sector_size'],
                                        z_order)
                    legend_handles.append(mpatches.Patch(color=psc[1],
                                            label=psc[0]))

        ax.legend(handles=legend_handles,fontsize=10,title="$\\bf{Protection\ standard\ (years)}$",loc='lower left')
        ax.text(
            0.05,
            0.95,
            figure_texts[j+4],
            horizontalalignment='left',
            transform=ax.transAxes,
            size=18,
            weight='bold')     

        j+=1            

    plt.tight_layout()
    save_fig(os.path.join(figures,'flood_protection_investments.png'))
    plt.close()


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
