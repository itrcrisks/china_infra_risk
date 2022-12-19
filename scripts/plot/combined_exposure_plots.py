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
from map_plotting_utils import (load_config,get_projection,
                            plot_basemap, plot_point_assets, 
                            plot_line_assets, save_fig)
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

def main(config):
    data_path = config['paths']['data']
    output_path = config['paths']['output']
    figure_path = config['paths']['figures']
    network_path = os.path.join(data_path,'networks') # Where we have all the network shapefiles
    asset_details_path = os.path.join(data_path,'asset_details') # Where we have all the exposure results
    risk_results_path = os.path.join(output_path,'risk_and_adaptation_results') # Where we have all the risk results

    figures = os.path.join(figure_path,'exposures')
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

    plot_types = [
                        {
                        'type':'exposures',
                        'groupby':[
                                    'rcp',
                                    'epoch',
                                    'rp'
                                ],
                        'file_name':'exposures_numbers_climate_scenario_year_return_period.xlsx',
                        'plot_name':'exposures_numbers_current_return_period_maps.png'
                        },
                        {
                        'type':'exposures',
                        'groupby':[
                                    'rcp',
                                    'epoch',
                                    'rp'
                                ],
                        'file_name':'exposures_numbers_climate_scenario_year_return_period.xlsx',
                        'plot_name':'exposures_numbers_current_return_period.png'
                        },
                        {
                        'type':'exposures',
                        'groupby':[
                                    'rcp',
                                    'epoch',
                                    'rp'
                                ],
                        'file_name':'exposures_numbers_climate_scenario_year_return_period.xlsx',
                        'plot_name':'exposures_numbers_current_return_period_lineplots.png'
                        },
                        {
                        'type':'exposures',
                        'groupby':[
                                    'rcp',
                                    'epoch',
                                    'rp'
                                ],
                        'years':[2030,2050,2080],
                        'climate_scenarios':['4.5','8.5'], 
                        'scenario_color':['#d94801','#7f2704'], 
                        'scenario_marker':['s-','^-'],     
                        'file_name':'exposures_numbers_climate_scenario_year_return_period.xlsx',
                        'plot_name':'exposures_numbers_climate_scenario_year_return_period.png'
                        }
                    ]

    baseyear = 1980
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
    for st in range(len(plot_types)):
        if st == 0:
            figure_texts = ['a.','b.','c.','d.']
            fig, ax_plots = plt.subplots(2,2,
                    subplot_kw={'projection': ax_proj},
                    figsize=(10,10),
                    dpi=500)
            ax_plots = ax_plots.flatten()
            j = 0
            for s in range(len(sector_descriptions)):
                legend_handles = []
                sector = sector_descriptions[s]
                assets = gpd.read_file(os.path.join(network_path,
                                    sector['sector_shapefile']),layer=sector['asset_type'])
                
                exposures = pd.read_parquet(os.path.join(risk_results_path,sector['sector'],
                        f"{sector['sector']}_{sector['asset_type']}_{sector['flood_protection_column']}_exposures.parquet")) 
                flood_ids = list(set(exposures[sector['id_column']].values.tolist()))
                del exposures
                ax = plot_basemap(ax_plots[j],include_labels=False)
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
                    figure_texts[j],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=14,
                    weight='bold')
                
                ax.set_title(
                            sector['sector_label'],
                            fontdict = {'fontsize':18,
                            'fontweight':'bold'})
                j+=1
            plt.tight_layout()
            save_fig(os.path.join(figures,plot_types[st]['plot_name']))
            plt.close()
        elif st == 1:
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
                
                exposures = pd.read_parquet(os.path.join(risk_results_path,sector['sector'],
                        f"{sector['sector']}_{sector['asset_type']}_{sector['flood_protection_column']}_exposures.parquet")) 
                flood_ids = list(set(exposures[sector['id_column']].values.tolist()))
                del exposures
                ax = plot_basemap(ax_plots[j],include_labels=False)
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
                    figure_texts[j],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=14,
                    weight='bold')
                # ax.text(
                #     0.35,
                #     0.95,
                #     sector['sector_label'],
                #     horizontalalignment='left',
                #     transform=ax.transAxes,
                #     size=14,
                #     weight='bold')
                ax.set_title(
                            sector['sector_label'],
                            fontdict = {'fontsize':18,
                            'fontweight':'bold'})
                ax_plots[j+4].remove()
                ax = fig.add_subplot(2, 4, j+5)
                
                exposures_no_protection_rp = pd.read_excel(os.path.join(output_path,
                                        'exposures',
                                        f"{sector['sector']}_{plot_types[st]['file_name']}"),
                            sheet_name=f"{sector['sector']}-no_protection_rp")[
                            plot_types[st]['groupby'] + [f"{sector['exposure_column']}_mean"]]
                exposures_no_protection_rp.rename(columns={f"{sector['exposure_column']}_mean":'no_protection_rp'},inplace=True)
                exposures_protected = pd.read_excel(os.path.join(output_path,
                                        'exposures',
                                        f"{sector['sector']}_{plot_types[st]['file_name']}"),
                            sheet_name=f"{sector['sector']}-design_protection_rp")[
                            plot_types[st]['groupby'] + [f"{sector['exposure_column']}_mean"]]
                exposures_protected.rename(columns={f"{sector['exposure_column']}_mean":'protected'},inplace=True)

                exposures = pd.merge(exposures_no_protection_rp,exposures_protected,how='left',on=plot_types[st]['groupby']).fillna(0)
                # print (exposures)

                ax.plot(exposures[exposures['epoch'] == baseyear]['rp'],
                        sector['length_unit']*exposures[exposures['epoch'] == baseyear]['no_protection_rp'],
                        'x-',color='blue',
                        label='No protection')
                ax.plot(exposures[exposures['epoch'] == baseyear]['rp'],
                        sector['length_unit']*exposures[exposures['epoch'] == baseyear]['protected'],
                        'o-',color='orange',
                        label='Existing protection')
                ax.set_xlabel('Return period (years)',fontsize=14,fontweight='bold')
                if sector['asset_type'] == 'nodes':
                    ax.set_ylabel('Flooded number of assets',fontsize=14,fontweight='bold')
                else:
                    ax.set_ylabel('Flooded length (km)',fontsize=14,fontweight='bold')
                ax.legend(fontsize=12,loc='lower right')
                ax.set_xscale('log')
                ax.tick_params(axis='both', labelsize=11)
                ax.set_xticks([t for t in list(set(exposures[exposures['epoch'] == baseyear]['rp'].values))])
                ax.set_xticklabels([str(t) for t in list(set(exposures[exposures['epoch'] == baseyear]['rp'].values))])
                ax.grid(True)
                # ax.set_xticks([t for t in list(set(exposures[exposures['year'] == baseyear]['return_period'].values))], 
                #             [str(t) for t in list(set(exposures[exposures['year'] == baseyear]['return_period'].values))])
                ax.text(
                    0.05,
                    0.95,
                    figure_texts[j+4],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=14,
                    weight='bold')

                j+=1            

            plt.tight_layout()
            save_fig(os.path.join(figures,plot_types[st]['plot_name']))
            plt.close()

        elif st == 2:
            figure_texts = ['a.','b.','c.','d.']
            fig, ax_plots = plt.subplots(2,2,
                    figsize=(12,12),
                    dpi=500)
            ax_plots = ax_plots.flatten()
            j = 0
            for s in range(len(sector_descriptions)):
                sector = sector_descriptions[s]
                ax = ax_plots[j]
                exposures_no_protection_rp = pd.read_excel(os.path.join(output_path,
                                        'exposures',
                                        f"{sector['sector']}_{plot_types[st]['file_name']}"),
                            sheet_name=f"{sector['sector']}-no_protection_rp")[
                            plot_types[st]['groupby'] + [f"{sector['exposure_column']}_mean"]]
                exposures_no_protection_rp.rename(columns={f"{sector['exposure_column']}_mean":'no_protection_rp'},inplace=True)
                exposures_protected = pd.read_excel(os.path.join(output_path,
                                        'exposures',
                                        f"{sector['sector']}_{plot_types[st]['file_name']}"),
                            sheet_name=f"{sector['sector']}-design_protection_rp")[
                            plot_types[st]['groupby'] + [f"{sector['exposure_column']}_mean"]]
                exposures_protected.rename(columns={f"{sector['exposure_column']}_mean":'protected'},inplace=True)

                exposures = pd.merge(exposures_no_protection_rp,exposures_protected,how='left',on=plot_types[st]['groupby']).fillna(0)
                # print (exposures)

                ax.plot(exposures[exposures['epoch'] == baseyear]['rp'],
                        sector['length_unit']*exposures[exposures['epoch'] == baseyear]['no_protection_rp'],
                        'x-',color='blue',
                        label='No protection')
                ax.plot(exposures[exposures['epoch'] == baseyear]['rp'],
                        sector['length_unit']*exposures[exposures['epoch'] == baseyear]['protected'],
                        'o-',color='orange',
                        label='Existing protection')
                ax.set_xlabel('Return period (years)',fontsize=14,fontweight='bold')
                if sector['asset_type'] == 'nodes':
                    ax.set_ylabel('Flooded number of assets',fontsize=14,fontweight='bold')
                else:
                    ax.set_ylabel('Flooded length (km)',fontsize=14,fontweight='bold')
                ax.legend(loc='lower right',prop={'size':12,'weight':'bold'})
                ax.set_xscale('log')
                ax.tick_params(axis='both', labelsize=11)
                ax.set_xticks([t for t in list(set(exposures[exposures['epoch'] == baseyear]['rp'].values))])
                ax.set_xticklabels([str(t) for t in list(set(exposures[exposures['epoch'] == baseyear]['rp'].values))])
                ax.grid(True)
                # ax.set_xticks([t for t in list(set(exposures[exposures['year'] == baseyear]['return_period'].values))], 
                #             [str(t) for t in list(set(exposures[exposures['year'] == baseyear]['return_period'].values))])
                ax.text(
                    0.05,
                    0.95,
                    figure_texts[j],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=18,
                    weight='bold')

                ax.text(
                    0.35,
                    0.95,
                    sector['sector_label'],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=18,
                    weight='bold')

                j+=1            

            plt.tight_layout()
            save_fig(os.path.join(figures,plot_types[st]['plot_name']))
            plt.close()
    
        elif st == 3:
            figure_texts = ['a.','b.','c.','d.','e.','f.','g.','h.','i.','j.','k.','l.']
            quantiles_list = ['mean','amin','amax']
            fig, ax_plots = plt.subplots(3,4,
                    figsize=(30,16),
                    dpi=500)
            ax_plots = ax_plots.flatten()
            j = 0
            for year in plot_types[st]['years']:
                for s in range(len(sector_descriptions)):
                    sector = sector_descriptions[s]
                    exposures = pd.read_excel(os.path.join(output_path,
                                            'exposures',
                                            f"{sector['sector']}_{plot_types[st]['file_name']}"),
                                sheet_name=f"{sector['sector']}-design_protection_rp")[
                                plot_types[st]['groupby'] + [f"{sector['exposure_column']}_{g}" for g in quantiles_list]]
                    rps = list(set(exposures['rp'].values)) 
                    ax = ax_plots[j]
                    ax.plot(exposures[exposures['epoch'] == baseyear]['rp'],
                            sector['length_unit']*exposures[exposures['epoch'] == baseyear][f"{sector['exposure_column']}_{quantiles_list[0]}"],
                            'o-',color='#fd8d3c',markersize=10,linewidth=2.0,
                            label='Baseline')
                    for c in range(len(plot_types[st]['climate_scenarios'])):
                        sc = plot_types[st]['climate_scenarios'][c]
                        cl = plot_types[st]['scenario_color'][c]
                        m = plot_types[st]['scenario_marker'][c]
                        exp = exposures[(exposures['epoch'] == year) & (exposures['rcp'] == sc)]
                        ax.plot(exp['rp'],
                                sector['length_unit']*exp[f"{sector['exposure_column']}_{quantiles_list[0]}"],
                                m,color=cl,markersize=10,linewidth=2.0,
                                label=f"{sc.upper()} mean")
                        ax.fill_between(exp['rp'],sector['length_unit']*exp[f"{sector['exposure_column']}_{quantiles_list[0]}"],
                            sector['length_unit']*exp[f"{sector['exposure_column']}_{quantiles_list[1]}"],alpha=0.3,facecolor=cl)
                        ax.fill_between(exp['rp'],sector['length_unit']*exp[f"{sector['exposure_column']}_{quantiles_list[0]}"],
                            sector['length_unit']*exp[f"{sector['exposure_column']}_{quantiles_list[2]}"],alpha=0.3,facecolor=cl,
                            label=f"{sc.upper()} min-max")


                    
                    ax.set_xlabel('Return period (years)',fontsize=14,fontweight='bold')
                    if sector['asset_type'] == 'nodes':
                        ax.set_ylabel('Flooded number of assets',fontsize=14,fontweight='bold')
                    else:
                        ax.set_ylabel('Flooded length (km)',fontsize=14,fontweight='bold')
                    # if j == 9:    
                    #     ax.legend(loc='lower right',prop={'size':14,'weight':'bold'})
                    ax.set_xscale('log')
                    # if sector['sector'] == 'road':
                    #     ax.set_yscale('log')

                    ax.tick_params(axis='both', labelsize=14)
                    ax.set_xticks([t for t in rps])
                    ax.set_xticklabels([str(t) for t in rps])
                    ax.grid(True)
                    # ax.set_xticks([t for t in list(set(exposures[exposures['year'] == baseyear]['return_period'].values))], 
                    #             [str(t) for t in list(set(exposures[exposures['year'] == baseyear]['return_period'].values))])
                    ax.text(
                        0.05,
                        0.95,
                        figure_texts[j],
                        horizontalalignment='left',
                        transform=ax.transAxes,
                        size=18,
                        weight='bold')
                    ax.text(
                        0.05,
                        0.80,
                        year,
                        horizontalalignment='left',
                        transform=ax.transAxes,
                        size=18,
                        weight='bold')
                    if j <= 3:
                        # ax.text(
                        #     0.35,
                        #     0.95,
                        #     sector['sector_label'],
                        #     horizontalalignment='left',
                        #     transform=ax.transAxes,
                        #     size=18,
                        #     weight='bold')
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


if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
