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
    risk_results_path = os.path.join(output_path,'risk_and_adaptation_results') # Where we have all the risk results

    figures = os.path.join(figure_path,'exposures_damages_losses')
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
                        'type':['exposures','damages','losses'],
                        'groupby':[
                                    'rcp',
                                    'epoch',
                                    'rp'
                                ],
                        'file_name':'numbers_climate_scenario_year_return_period.xlsx',
                        'plot_name':'exposures_damage_loss_numbers_current_return_period_lineplots.png'
                        },
                        {
                        'type':'damages',
                        'groupby':[
                                    'rcp',
                                    'epoch',
                                    'rp'
                                ],
                        'years':[2030,2050,2080],
                        'climate_scenarios':['4.5','8.5'], 
                        'scenario_color':['#d94801','#7f2704'], 
                        'scenario_marker':['s-','^-'],     
                        'file_name':'damages_numbers_climate_scenario_year_return_period.xlsx',
                        'plot_name':'damages_numbers_climate_scenario_year_return_period.png'
                        },
                        {
                        'type':'losses',
                        'groupby':[
                                    'rcp',
                                    'epoch',
                                    'rp'
                                ],
                        'years':[2030,2050,2080],
                        'climate_scenarios':['4.5','8.5'], 
                        'scenario_color':['#d94801','#7f2704'], 
                        'scenario_marker':['s-','^-'],     
                        'file_name':'losses_numbers_climate_scenario_year_return_period.xlsx',
                        'plot_name':'losses_numbers_climate_scenario_year_return_period.png'
                        },
                        {
                        'type':'damages_plus_losses',
                        'groupby':[
                                    'rcp',
                                    'epoch',
                                    'rp'
                                ],
                        'years':[2030,2050,2080],
                        'climate_scenarios':['4.5','8.5'], 
                        'scenario_color':['#d94801','#7f2704'], 
                        'scenario_marker':['s-','^-'],     
                        'file_name':'numbers_climate_scenario_year_return_period.xlsx',
                        'plot_name':'damages_plus_losses_numbers_climate_scenario_year_return_period.png'
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
    for st in range(len(plot_types)):
        if st == 0:
            figure_texts = ['a.','b.','c.','d.','e.','f.','g.','h.','i.','j.','k.','l.']
            fig, ax_plots = plt.subplots(3,4,
                    figsize=(30,16),
                    dpi=500)
            ax_plots = ax_plots.flatten()
            j = 0
            for pt_type in plot_types[st]['type']:
                for s in range(len(sector_descriptions)):
                    sector = sector_descriptions[s]
                    ax = ax_plots[j]
                    ax.tick_params(axis='both', labelsize=11)
                    y_ticks = []
                    for idx,(protection,label,color,marker) in enumerate(
                                            [("no_protection_rp","No protection","blue","x-"),
                                            ("design_protection_rp","Existing protection","orange","o-")]
                                            ):
                        exposures = pd.read_excel(os.path.join(output_path,
                                                pt_type,
                                                f"{sector['sector']}_{pt_type}_{plot_types[st]['file_name']}"),
                                    sheet_name=f"{sector['sector']}-{protection}")
                        if pt_type == "exposures":
                            unit = sector['length_unit']
                        else:
                            unit = cost_unit

                        ax.plot(exposures[exposures['epoch'] == baseyear]['rp'],
                                unit*exposures[exposures['epoch'] == baseyear][f"{sector['exposure_column']}_mean"],
                                marker,color=color,
                                label=f'{label} mean')
                        if pt_type != "exposures":
                            ax.fill_between(exposures[exposures['epoch'] == baseyear]['rp'],
                                    unit*exposures[exposures['epoch'] == baseyear][f"{sector['exposure_column']}_amin"],
                                    unit*exposures[exposures['epoch'] == baseyear][f"{sector['exposure_column']}_amax"],
                                    alpha=0.3,facecolor=color,
                                    label=f"{label} min-max")
                            y_ticks.append(unit*exposures[exposures['epoch'] == baseyear][f"{sector['exposure_column']}_mean"].mean())
                            y_ticks.append(
                                unit*exposures[(
                                            exposures['epoch'] == baseyear
                                            ) & (
                                            exposures[f"{sector['exposure_column']}_amin"] > 0
                                            )][f"{sector['exposure_column']}_amin"].min())
                            y_ticks.append(unit*exposures[exposures['epoch'] == baseyear][f"{sector['exposure_column']}_amax"].max())

                    ax.set_xlabel('Return period (years)',fontsize=14,fontweight='bold')
                    if pt_type == "exposures" and sector['asset_type'] == 'nodes':
                        ax.set_ylabel('Flooded number of assets',fontsize=14,fontweight='bold')
                    elif pt_type == "exposures" and sector['asset_type'] == 'edges':
                        ax.set_ylabel('Flooded length (km)',fontsize=14,fontweight='bold')
                    else:
                        ax.set_ylabel(f"{str(pt_type).title()} (RMB billion)",fontsize=14,fontweight='bold')
                        mod_ytick = []
                        for t in y_ticks:
                            if t == 0:
                                mod_ytick.append(1e-2)
                            elif t < 1:
                                mod_ytick.append(round(t,2))
                            else:
                                mod_ytick.append(int(t))

                        y_ticks = sorted(list(set(mod_ytick)))
                        print (y_ticks)
                        ax.set_yscale('log')
                        ax.set_yticks(y_ticks)
                        ax.set_yticklabels([str(t) for t in y_ticks])
                        # ax.set_yscale('log')

                    ax.set_xscale('log')
                    ax.set_xticks([t for t in list(set(exposures[exposures['epoch'] == baseyear]['rp'].values))])
                    ax.set_xticklabels([str(t) for t in list(set(exposures[exposures['epoch'] == baseyear]['rp'].values))])
                    ax.grid(True)
                    ax.text(
                        0.05,
                        0.95,
                        figure_texts[j],
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
                        title='Baseline estimates',
                        title_fontproperties={'size':20,'weight':'bold'},
                        prop={'size':18,'weight':'bold'})
            plt.tight_layout()
            save_fig(os.path.join(figures,plot_types[st]['plot_name']))
            plt.close()
    
        elif st in (1,2):
            figure_texts = ['a.','b.','c.','d.','e.','f.','g.','h.','i.','j.','k.','l.']
            quantiles_list = ['mean','amin','amax']
            fig, ax_plots = plt.subplots(3,4,
                    figsize=(30,16),
                    dpi=500)
            ax_plots = ax_plots.flatten()
            j = 0
            for year in plot_types[st]['years']:
                for s in range(len(sector_descriptions)):
                    y_ticks = []
                    sector = sector_descriptions[s]
                    exposures = pd.read_excel(os.path.join(output_path,
                                            plot_types[st]['type'],
                                            f"{sector['sector']}_{plot_types[st]['file_name']}"),
                                sheet_name=f"{sector['sector']}-design_protection_rp")[
                                plot_types[st]['groupby'] + [f"{sector['exposure_column']}_{g}" for g in quantiles_list]]
                    rps = list(set(exposures['rp'].values)) 
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_mean"] > 0
                                    )][f"{sector['exposure_column']}_mean"].min())
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_mean"] > 0
                                    )][f"{sector['exposure_column']}_mean"].max())
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_amin"] > 0
                                    )][f"{sector['exposure_column']}_amin"].min())
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_amin"] > 0
                                    )][f"{sector['exposure_column']}_amin"].max())
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_amax"] > 0
                                    )][f"{sector['exposure_column']}_amax"].min())
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_amax"] > 0
                                    )][f"{sector['exposure_column']}_amax"].max())
                    ax = ax_plots[j]
                    base_exp = exposures[exposures['epoch'] == baseyear]
                    ax.plot(base_exp['rp'],
                            cost_unit*base_exp[f"{sector['exposure_column']}_{quantiles_list[0]}"],
                            'o-',color='#fd8d3c',markersize=10,linewidth=2.0,
                            label='Baseline mean')
                    ax.fill_between(base_exp['rp'],
                            cost_unit*base_exp[f"{sector['exposure_column']}_{quantiles_list[1]}"],
                            cost_unit*base_exp[f"{sector['exposure_column']}_{quantiles_list[2]}"],
                            alpha=0.3,facecolor='#fd8d3c',
                            label=f"Baseline min-max")

                    # y_ticks.append(cost_unit*base_exp[f"{sector['exposure_column']}_mean"].mean())
                    # y_ticks.append(
                    #     cost_unit*base_exp[(
                    #                 base_exp[f"{sector['exposure_column']}_amin"] > 0
                    #                 )][f"{sector['exposure_column']}_amin"].min())
                    # y_ticks.append(cost_unit*base_exp[f"{sector['exposure_column']}_amax"].max())
                    for c in range(len(plot_types[st]['climate_scenarios'])):
                        sc = plot_types[st]['climate_scenarios'][c]
                        cl = plot_types[st]['scenario_color'][c]
                        m = plot_types[st]['scenario_marker'][c]
                        exp = exposures[(exposures['epoch'] == year) & (exposures['rcp'] == sc)]
                        ax.plot(exp['rp'],
                                cost_unit*exp[f"{sector['exposure_column']}_{quantiles_list[0]}"],
                                m,color=cl,markersize=10,linewidth=2.0,
                                label=f"RCP {sc.upper()} mean")
                        ax.fill_between(exp['rp'],cost_unit*exp[f"{sector['exposure_column']}_{quantiles_list[1]}"],
                            cost_unit*exp[f"{sector['exposure_column']}_{quantiles_list[2]}"],alpha=0.3,facecolor=cl,
                            label=f"RCP {sc.upper()} min-max")


                        # y_ticks.append(cost_unit*exp[f"{sector['exposure_column']}_mean"].mean())
                        # y_ticks.append(
                        #     cost_unit*exp[(
                        #                 exp[f"{sector['exposure_column']}_amin"] > 0
                        #                 )][f"{sector['exposure_column']}_amin"].min())
                        # y_ticks.append(cost_unit*exp[f"{sector['exposure_column']}_amax"].max())
                    
                    ax.set_xlabel('Return period (years)',fontsize=14,fontweight='bold')
                    ax.set_ylabel(f"{str(plot_types[st]['type']).title()} (RMB billion)",fontsize=14,fontweight='bold')
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    mod_ytick = []
                    for t in y_ticks:
                        if round (t,2) == 0:
                            mod_ytick.append(1e-2)
                        elif t < 1:
                            mod_ytick.append(round(t,2))
                        else:
                            mod_ytick.append(int(t))

                    y_ticks = sorted(list(set(mod_ytick)))
                    print (y_ticks)
                    # ax.set_yscale('log')
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels([str(t) for t in y_ticks])

                    ax.tick_params(axis='both', labelsize=14)
                    ax.set_xticks([t for t in rps])
                    ax.set_xticklabels([str(t) for t in rps])
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
                        0.05,
                        0.80,
                        year,
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
                    y_ticks = []
                    sector = sector_descriptions[s]
                    damages = pd.read_excel(os.path.join(output_path,
                                            'damages',
                                            f"{sector['sector']}_damages_{plot_types[st]['file_name']}"),
                                sheet_name=f"{sector['sector']}-design_protection_rp")[
                                plot_types[st]['groupby'] + [f"{sector['exposure_column']}_{g}" for g in quantiles_list]]
                    damages.rename(columns = dict([(f"{sector['exposure_column']}_{g}",f"damages_{g}") for g in quantiles_list]),inplace=True)
                    # print (damages)
                    losses = pd.read_excel(os.path.join(output_path,
                                            'losses',
                                            f"{sector['sector']}_losses_{plot_types[st]['file_name']}"),
                                sheet_name=f"{sector['sector']}-design_protection_rp")[
                                plot_types[st]['groupby'] + [f"{sector['exposure_column']}_{g}" for g in quantiles_list]]
                    losses.rename(columns = dict([(f"{sector['exposure_column']}_{g}",f"losses_{g}") for g in quantiles_list]),inplace=True)
                    # print (losses)
                    exposures = pd.merge(damages,losses,how="left",on=plot_types[st]['groupby'])
                    for g in quantiles_list:
                        exposures[f"{sector['exposure_column']}_{g}"] = exposures[f"damages_{g}"] + exposures[f"losses_{g}"]
                    rps = list(set(exposures['rp'].values)) 
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_mean"] > 0
                                    )][f"{sector['exposure_column']}_mean"].min())
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_mean"] > 0
                                    )][f"{sector['exposure_column']}_mean"].max())
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_amin"] > 0
                                    )][f"{sector['exposure_column']}_amin"].min())
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_amin"] > 0
                                    )][f"{sector['exposure_column']}_amin"].max())
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_amax"] > 0
                                    )][f"{sector['exposure_column']}_amax"].min())
                    y_ticks.append(
                        cost_unit*exposures[(
                                    exposures[f"{sector['exposure_column']}_amax"] > 0
                                    )][f"{sector['exposure_column']}_amax"].max())
                    ax = ax_plots[j]
                    base_exp = exposures[exposures['epoch'] == baseyear]
                    ax.plot(base_exp['rp'],
                            cost_unit*base_exp[f"{sector['exposure_column']}_{quantiles_list[0]}"],
                            'o-',color='#fd8d3c',markersize=10,linewidth=2.0,
                            label='Baseline mean')
                    ax.fill_between(base_exp['rp'],
                            cost_unit*base_exp[f"{sector['exposure_column']}_{quantiles_list[1]}"],
                            cost_unit*base_exp[f"{sector['exposure_column']}_{quantiles_list[2]}"],
                            alpha=0.3,facecolor='#fd8d3c',
                            label=f"Baseline min-max")

                    # y_ticks.append(cost_unit*base_exp[f"{sector['exposure_column']}_mean"].mean())
                    # y_ticks.append(
                    #     cost_unit*base_exp[(
                    #                 base_exp[f"{sector['exposure_column']}_amin"] > 0
                    #                 )][f"{sector['exposure_column']}_amin"].min())
                    # y_ticks.append(cost_unit*base_exp[f"{sector['exposure_column']}_amax"].max())
                    for c in range(len(plot_types[st]['climate_scenarios'])):
                        sc = plot_types[st]['climate_scenarios'][c]
                        cl = plot_types[st]['scenario_color'][c]
                        m = plot_types[st]['scenario_marker'][c]
                        exp = exposures[(exposures['epoch'] == year) & (exposures['rcp'] == sc)]
                        ax.plot(exp['rp'],
                                cost_unit*exp[f"{sector['exposure_column']}_{quantiles_list[0]}"],
                                m,color=cl,markersize=10,linewidth=2.0,
                                label=f"RCP {sc.upper()} mean")
                        ax.fill_between(exp['rp'],cost_unit*exp[f"{sector['exposure_column']}_{quantiles_list[1]}"],
                            cost_unit*exp[f"{sector['exposure_column']}_{quantiles_list[2]}"],alpha=0.3,facecolor=cl,
                            label=f"RCP {sc.upper()} min-max")


                        # y_ticks.append(cost_unit*exp[f"{sector['exposure_column']}_mean"].mean())
                        # y_ticks.append(
                        #     cost_unit*exp[(
                        #                 exp[f"{sector['exposure_column']}_amin"] > 0
                        #                 )][f"{sector['exposure_column']}_amin"].min())
                        # y_ticks.append(cost_unit*exp[f"{sector['exposure_column']}_amax"].max())
                    
                    ax.set_xlabel('Return period (years)',fontsize=14,fontweight='bold')
                    ax.set_ylabel(f"Damages + Losses (RMB billion)",fontsize=14,fontweight='bold')
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    mod_ytick = []
                    for t in y_ticks:
                        if round (t,2) == 0:
                            mod_ytick.append(1e-2)
                        elif t < 1:
                            mod_ytick.append(round(t,2))
                        else:
                            mod_ytick.append(int(t))

                    y_ticks = sorted(list(set(mod_ytick)))
                    print (y_ticks)
                    # ax.set_yscale('log')
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels([str(t) for t in y_ticks])

                    ax.tick_params(axis='both', labelsize=14)
                    ax.set_xticks([t for t in rps])
                    ax.set_xticklabels([str(t) for t in rps])
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
                        0.05,
                        0.80,
                        year,
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
        elif st == 4:
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
                                label=f"RCP {sc.upper()} mean")
                        # ax.fill_between(exp['rp'],sector['length_unit']*exp[f"{sector['exposure_column']}_{quantiles_list[0]}"],
                        #     sector['length_unit']*exp[f"{sector['exposure_column']}_{quantiles_list[1]}"],alpha=0.3,facecolor=cl)
                        ax.fill_between(exp['rp'],sector['length_unit']*exp[f"{sector['exposure_column']}_{quantiles_list[1]}"],
                            sector['length_unit']*exp[f"{sector['exposure_column']}_{quantiles_list[2]}"],alpha=0.3,facecolor=cl,
                            label=f"RCP {sc.upper()} min-max")


                    
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
