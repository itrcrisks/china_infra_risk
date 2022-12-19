"""Risk plots
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
from map_plotting_utils import (load_config, get_projection, 
                                plot_basemap, plot_point_assets,
                                point_map_plotting_color_width,
                                plot_line_assets,line_map_plotting_colors_width,
                                save_fig,line_map_plotting,point_map_plotting)
from tqdm import tqdm
tqdm.pandas()

def main(config):
    data_path = config['paths']['data']
    output_path = config['paths']['output']
    figure_path = config['paths']['figures']
    network_path = os.path.join(data_path,'networks') # Where we have all the network shapefiles
    asset_details_path = os.path.join(output_path,'risks_timeseries_assets') # Where we have all the exposure results
    risk_results_path = os.path.join(output_path,'risk_and_adaptation_results') # Where we have all the risk results

    figures = os.path.join(figure_path,'risks')
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
                    # {
                    # 'type':'asset_risks',
                    # # 'EAD_dropby':[
                    # #                 'flood_parameter',
                    # #                 'fragility_parameter',
                    # #             ],

                    # # 'EAEL_dropby':[
                    # #                 'duration',
                    # #                 'economic_parameter',
                    # #                 ],
                    # 'plot_name':'mean_risks_baseline_year'
                    # },
                    # {
                    # 'type':'asset_risks_future',
                    # 'years':[2030,2050,2080],
                    # 'climate_scenarios':[4.5,8.5],
                    # 'plot_name':'mean_risks_futures_climate_scenarios'
                    # },
                    {
                    'type':'asset_risks_changes',
                    'years':[2030,2050,2080],
                    'climate_scenarios':[4.5,8.5],
                    'plot_name':'mean_risks_changes_climate_scenarios'
                    },
                    {
                    'type':'risk_timeseries_lineplots',
                    'groupby':[
                                'rcp',
                                'epoch'
                            ],
                    'climate_scenarios':[4.5,8.5], 
                    'scenario_color':['#2171b5','#08306b'], 
                    'scenario_marker':['s-','^-'],     
                    'plot_name':'risk_timeseries_lineplots'
                    },
                ]
    baseyear = 2020
    flood_color = '#3182bd'
    noflood_color = '#969696'
    flood_colors = ['#9ecae1','#6baed6','#2171b5','#08306b']
    flood_return_periods = [2,5,10,25,50,100,250,500,1000]
    protection_standards_details = [
                                    {'standard':'50',
                                    'standard_name':'_50_year',
                                    'color':'#fb6a4a',
                                    'marker':'^-',
                                    'label':'50-year protection'
                                    },
                                    {'standard':'100',
                                    'standard_name':'_100_year',
                                    'color':'#cb181d',
                                    'marker':'<-',
                                    'label':'100-year protection'
                                    },
                                    {'standard':'250',
                                    'standard_name':'_250_year',
                                    'color':'#a50f15',
                                    'marker':'>-',
                                    'label':'250-year protection'
                                    },
                                    {'standard':'500',
                                    'standard_name':'_500_year',
                                    'color':'#67000d',
                                    'marker':'p-',
                                    'label':'500-year protection'
                                    },
                                    {'standard':'1000',
                                    'standard_name':'_1000_year',
                                    'color':'#525252',
                                    'marker':'*-',
                                    'label':'1000-year protection'
                                    },
                                    {'standard':'designed_protection',
                                    'standard_name':'designed_protection',
                                    'color':'#ef3b2c',
                                    'marker':'o-',
                                    'label':'Existing protection'
                                    },
                                    {'standard':'undefended',
                                    'standard_name':'undefended',
                                    'color':'#000000',
                                    'marker':'s-',
                                    'label':'No protection'
                                    }
                                ]
    #fff5eb
    #fee6ce
    #fdd0a2
    #fdae6b
    #fd8d3c
    #f16913
    #d94801
    #a63603
    #7f2704

    #f7fbff
    #deebf7
    #c6dbef
    #9ecae1
    #6baed6
    #4292c6
    #2171b5
    #08519c
    #08306b
    division_factor = 1e6

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
    axis_log_scale = False
    include_undefended = False
    # cases = ['undefended','designed_protection','avoided']
    cases = ['design_protection_rp']
    risk_types = ['EAD','EAEL','total']
    # risk_types = ['EAD','EAEL']
    risk_types_label = ['EAD','EAL','Total Risk']
    if len(risk_types) == 3:
        plt_ht = 16
    else:
        plt_ht = 10
    for case in cases:
        # risk_columns = [f"EAD_{case}",f"EAEL_{case}"]
        for st in range(len(plot_types)):
            if plot_types[st]['type'] == 'asset_risks':
                # plot_types[st]['total_dropby'] = plot_types[st]['EAD_dropby'] + plot_types[st]['EAEL_dropby']
                figure_texts = ['a.','b.','c.','d.','e.','f.','g.','h.','i.','j.','k.','l.']
                fig, ax_plots = plt.subplots(len(risk_types),len(sector_descriptions),
                        subplot_kw={'projection': ax_proj},
                        figsize=(24,plt_ht),
                        dpi=500)
                ax_plots = ax_plots.flatten()
                j = 0
                for s in range(len(sector_descriptions)):
                    sector = sector_descriptions[s]
                    sector_geom = gpd.read_file(os.path.join(network_path,
                                    sector['sector_shapefile']),layer=sector['asset_type'])

                    # risks = risks[risks['year'] == baseyear]
                    risk_med = []
                    for rt in range(len(risk_types)):
                        if risk_types[rt] != 'total':
                            risks = pd.read_excel(os.path.join(asset_details_path,
                                        f"{sector['sector']}_asset_risk_timeseries_climate_scenarios_mean.xlsx"),
                                        sheet_name=f"{sector['sector']}-{risk_types[rt]}-design")
                            risks_df = risks[[sector['id_column'],str(baseyear)]].drop_duplicates(subset=[sector['id_column']],keep='first')
                            risks_df.rename(columns={str(baseyear):f"{risk_types[rt]}_{case}"},inplace=True)
                            if len(risk_med) > 0:
                                risk_med = pd.merge(risk_med,
                                                        risks_df[[sector['id_column'],f"{risk_types[rt]}_{case}"]],
                                                        how='left',
                                                        on=[sector['id_column']])
                            else:
                                risk_med = risks_df[[sector['id_column'],f"{risk_types[rt]}_{case}"]].copy()

                        else:
                            # risks_med = risk_med.copy()
                            risk_med[f"total_{case}"] = risk_med[f"EAD_{case}"] + risk_med[f"EAEL_{case}"]
                            risks_df = risk_med.copy()


                        assets = gpd.GeoDataFrame(pd.merge(sector_geom,
                                    risks_df[[sector['id_column'],f"{risk_types[rt]}_{case}"]],
                                    how='left',on=[sector['id_column']]),
                                    geometry='geometry',crs='epsg:4326')

                        assets[f"{risk_types[rt]}_{case}"].fillna(0,inplace=True)
                        assets[f"{risk_types[rt]}_{case}"] = assets[f"{risk_types[rt]}_{case}"]/division_factor
                        assets[f"{risk_types[rt]}_{case}"][assets[f"{risk_types[rt]}_{case}"]<0]= 0

                        ax = plot_basemap(ax_plots[j + len(sector_descriptions)*rt],include_labels=False)
                        # print (j + len(sector_descriptions)*rt)
                        if sector['asset_type'] == 'nodes':
                            ax = point_map_plotting_color_width(ax,assets,f"{risk_types[rt]}_{case}",
                                    sector['sector_marker'],1.0,
                                    f"{risk_types_label[rt]} (RMB millions)",'risk/flooding',
                                    point_colors = flood_colors,
                                    no_value_color = '#969696',
                                    point_steps = len(flood_colors)+1
                                    )
                        else:
                            ax = line_map_plotting_colors_width(ax,assets,f"{risk_types[rt]}_{case}",
                                        1.0,
                                        f"{risk_types_label[rt]} (RMB millions)",
                                        'risk/flooding',
                                        line_colors = flood_colors,
                                        no_value_color = '#969696',
                                        line_steps = len(flood_colors)+1,
                                        width_step = 0.03,
                                        significance=0,
                                        interpolation='fisher-jenks')
                        ax.text(
                            0.05,
                            0.95,
                            f"{figure_texts[j + len(sector_descriptions)*rt]} Residual {risk_types_label[rt]}",
                            horizontalalignment='left',
                            transform=ax.transAxes,
                            size=18,
                            weight='bold')
                        
                        if rt == 0:
                            ax.set_title(
                                sector['sector_label'],
                                fontdict = {'fontsize':18,
                                'fontweight':'bold'})
                            

                    j+=1
                plt.tight_layout()
                if len(risk_types) == 2:
                    save_fig(os.path.join(figures,f"{plot_types[st]['plot_name']}_{case}_without_total.png"))
                else:
                    save_fig(os.path.join(figures,f"{plot_types[st]['plot_name']}_{case}.png"))
                plt.close()
            elif plot_types[st]['type'] == 'asset_risks_future':
                # plot_types[st]['total_dropby'] = plot_types[st]['EAD_dropby'] + plot_types[st]['EAEL_dropby']
                for scenario in plot_types[st]['climate_scenarios']:
                    figure_texts = ['a.','b.','c.','d.','e.','f.','g.','h.','i.','j.','k.','l.']
                    fig, ax_plots = plt.subplots(3,len(sector_descriptions),
                            subplot_kw={'projection': ax_proj},
                            figsize=(24,16),
                            dpi=500)
                    ax_plots = ax_plots.flatten()
                    j = 0
                    for s in range(len(sector_descriptions)):
                        sector = sector_descriptions[s]
                        sector_geom = gpd.read_file(os.path.join(network_path,
                                    sector['sector_shapefile']),layer=sector['asset_type'])
                        # risks = risks[risks['year'] == baseyear]
                        for rt in range(len(plot_types[st]['years'])):
                            climate_year = plot_types[st]['years'][rt]
                            risk_med = []
                            for risk_type in ["EAD","EAEL"]:
                                risks = pd.read_excel(os.path.join(asset_details_path,
                                            f"{sector['sector']}_asset_risk_timeseries_climate_scenarios_mean.xlsx"),
                                            sheet_name=f"{sector['sector']}-{risk_type}-design")
                                risks_df = risks[risks['rcp'] == scenario][[sector['id_column'],str(climate_year)]]
                                risks_df.rename(columns={str(climate_year):f"{risk_type}_{case}"},inplace=True)
                                if len(risk_med) > 0:
                                    risk_med = pd.merge(risk_med,
                                                            risks_df[[sector['id_column'],f"{risk_type}_{case}"]],
                                                            how='left',
                                                            on=[sector['id_column']])
                                else:
                                    risk_med = risks_df[[sector['id_column'],f"{risk_type}_{case}"]].copy()

                            # print (risk_med)    
                            risk_med[f"total_{case}"] = risk_med[f"EAD_{case}"] + risk_med[f"EAEL_{case}"]
                            risks_df = risk_med.copy()


                            assets = gpd.GeoDataFrame(pd.merge(sector_geom,
                                        risks_df[[sector['id_column'],f"total_{case}"]],
                                        how='left',on=[sector['id_column']]),
                                        geometry='geometry',crs='epsg:4326')

                            assets[f"total_{case}"].fillna(0,inplace=True)
                            assets[f"total_{case}"] = assets[f"total_{case}"]/division_factor
                            assets[f"total_{case}"][assets[f"total_{case}"]<0]= 0

                            ax = plot_basemap(ax_plots[j + len(sector_descriptions)*rt],include_labels=False)
                            # print (j + len(sector_descriptions)*rt)
                            if sector['asset_type'] == 'nodes':
                                ax = point_map_plotting_color_width(ax,assets,f"total_{case}",
                                        sector['sector_marker'],1.0,
                                        f"Total Risk (RMB millions)",'risk/flooding',
                                        point_colors = flood_colors,
                                        no_value_color = '#969696',
                                        point_steps = len(flood_colors)+1
                                        )
                            else:
                                # print (assets[f"{risk_types[rt]}_{case}"])
                                # assets[[sector['id_column'],f"{risk_types[rt]}_{case}"]].to_csv('test.csv')
                                ax = line_map_plotting_colors_width(ax,assets,f"total_{case}",
                                            1.0,
                                            f"Total Risk (RMB millions)",
                                            'risk/flooding',
                                            line_colors = flood_colors,
                                            no_value_color = '#969696',
                                            line_steps = len(flood_colors)+1,
                                            width_step = 0.03,
                                            significance=0,
                                            interpolation='fisher-jenks')
                            ax.text(
                                0.05,
                                0.95,
                                f"{figure_texts[j + len(sector_descriptions)*rt]} RCP {scenario} - {climate_year}",
                                horizontalalignment='left',
                                transform=ax.transAxes,
                                size=18,
                                weight='bold')

                            # ax.text(
                            #     0.07,
                            #     0.95,
                            #     f"Residual {risk_types_label[rt]}",
                            #     horizontalalignment='left',
                            #     transform=ax.transAxes,
                            #     size=18,
                            #     weight='bold')
                            
                            if rt == 0:
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
                    plt.tight_layout()
                    save_fig(os.path.join(figures,
                            f"{plot_types[st]['plot_name']}_{case}_{str(scenario).replace('.','').replace(' ','')}.png"))
                    plt.close()

            elif plot_types[st]['type'] == 'asset_risks_changes':
                # plot_types[st]['total_dropby'] = plot_types[st]['EAD_dropby'] + plot_types[st]['EAEL_dropby']
                for scenario in plot_types[st]['climate_scenarios']:
                    figure_texts = ['a.','b.','c.','d.','e.','f.','g.','h.','i.','j.','k.','l.']
                    fig, ax_plots = plt.subplots(3,len(sector_descriptions),
                            figsize=(30,16),
                            dpi=500)
                    ax_plots = ax_plots.flatten()
                    j = 0
                    for s in range(len(sector_descriptions)):
                        sector = sector_descriptions[s]
                        risk_med = []
                        all_years = [baseyear] + plot_types[st]['years']
                        risks_ead = pd.read_excel(os.path.join(asset_details_path,
                                    f"{sector['sector']}_asset_risk_timeseries_climate_scenarios_mean.xlsx"),
                                    sheet_name=f"{sector['sector']}-EAD-design")
                        risks_ead = risks_ead[risks_ead['rcp'] == scenario][[sector['id_column']] + [str(yr) for yr in all_years]]
                        risks_ead.rename(columns=dict([(str(yr),f"EAD_{yr}") for yr in all_years]),
                                            inplace=True)
                        risks_eael = pd.read_excel(os.path.join(asset_details_path,
                                    f"{sector['sector']}_asset_risk_timeseries_climate_scenarios_mean.xlsx"),
                                    sheet_name=f"{sector['sector']}-EAEL-design")
                        risks_eael = risks_eael[risks_eael['rcp'] == scenario][[sector['id_column']] + [str(yr) for yr in all_years]]
                        risks_eael.rename(columns=dict([(str(yr),f"EAEL_{yr}") for yr in all_years]),
                                            inplace=True)
                        risks_df = pd.merge(risks_ead,risks_eael,
                                            how='left',
                                            on=[sector['id_column']])
                        risks_df['total'] = 0
                        min_limits = []
                        max_limits = []
                        for yr in all_years:    
                            risks_df[f"total_{yr}"] = risks_df[f"EAD_{yr}"]/division_factor + risks_df[f"EAEL_{yr}"]/division_factor
                            risks_df['total'] += risks_df[f"total_{yr}"]
                            min_limits.append(risks_df[f"total_{yr}"].min())
                            max_limits.append(risks_df[f"total_{yr}"].max())

                        risks_df = risks_df[risks_df['total'] > 0]
                            
                        for rt in range(len(plot_types[st]['years'])):
                            climate_year = plot_types[st]['years'][rt]
                            ax = ax_plots[j + len(sector_descriptions)*rt]
                            ax.scatter(risks_df[f"total_{baseyear}"],
                                    risks_df[f"total_{climate_year}"],
                                    label=f"RCP {scenario} mean risks",
                                    marker='o',
                                    s=30,
                                    color='#3182bd',zorder=20)

                            ax.set_ylim(min(min_limits),1.1*max(max_limits))

                            # lims = [
                            #             np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                            #             np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                            #         ]

                            # lims = np.array(lims)
                            # ranges = [0,1,3,9,1e4]
                            # ranges_labels = ["0-1","1-3","3-9","> 9"]
                            # colors = ["#053061","#4393c3","#f4a582","#67001f"]
                            # for mf in range(len(ranges)-1): 
                            #     ax.fill_between(lims,ranges[mf]*lims,
                            #         ranges[mf+1]*lims,alpha=0.7,facecolor=colors[mf],label=ranges_labels[mf])
                            
                            # ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
                            

                            ax.set_xlabel('Baseline Total Risks (RMB millions)',fontsize=14,fontweight='bold')
                            ax.set_ylabel(f'Future Total Risks (RMB millions)',fontsize=14,fontweight='bold')

                            if axis_log_scale is True:
                                ax.set_yscale('log')
                            # ax.set_xscale('log')
                            # ax.set_yscale('log')
                            # ax.set_ylim(0,risks[f'total_{case}_max'].max())
                            ax.tick_params(axis='both', labelsize=14)
                            ax.grid(True)
                            ax.text(
                                0.10,
                                0.95,
                                f"{figure_texts[j + len(sector_descriptions)*rt]} {climate_year}",
                                horizontalalignment='left',
                                transform=ax.transAxes,
                                size=18,
                                weight='bold')

                            # ax.text(
                            #     0.07,
                            #     0.95,
                            #     f"Residual {risk_types_label[rt]}",
                            #     horizontalalignment='left',
                            #     transform=ax.transAxes,
                            #     size=18,
                            #     weight='bold')
                            
                            if rt == 0:
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
                    legend = ax_plots[(len(plot_types[st]['years']) - 1)*len(sector_descriptions) - 1].legend(
                                        loc='upper left', 
                                        title="Ratios of future to baseline risks",
                                        bbox_to_anchor=(1.05,0.8),
                                        prop={'size':18,'weight':'bold'})
                    plt.setp(legend.get_title(),fontsize=20,fontweight='bold')
                    plt.tight_layout()
                    if axis_log_scale is True:
                        save_fig(os.path.join(figures,
                            f"{plot_types[st]['plot_name']}_{case}_{str(scenario).replace('.','').replace(' ','')}_logscale.png"))
                    else:
                        save_fig(os.path.join(figures,
                            f"{plot_types[st]['plot_name']}_{case}_{str(scenario).replace('.','').replace(' ','')}.png"))
                    plt.close()
            elif plot_types[st]['type'] == 'risk_timeseries_lineplots':
                figure_texts = ['a.','b.','c.','d.','e.','f.','g.','h.','k.','l.','m.','n.']
                quantiles_list = ['mean','amin','amax']
                fig, ax_plots = plt.subplots(len(risk_types),len(sector_descriptions),
                        figsize=(30,plt_ht),
                        dpi=500)
                ax_plots = ax_plots.flatten()
                j = 0
                for s in range(len(sector_descriptions)):
                    sector = sector_descriptions[s]
                    risks_ead = pd.read_excel(os.path.join(output_path,
                                        'risks_timeseries',
                                        f"{sector['sector']}_risk_timeseries_climate_scenario_year.xlsx"),
                            sheet_name=f"{sector['sector']}-EAD-design")[
                            plot_types[st]['groupby'] + [
                                                f"EAD_{case}_timeseries_npv_{g}" for g in quantiles_list]]
                    risks_eael = pd.read_excel(os.path.join(output_path,
                                        'risks_timeseries',
                                        f"{sector['sector']}_risk_timeseries_climate_scenario_year.xlsx"),
                            sheet_name=f"{sector['sector']}-EAEL-design")[
                            plot_types[st]['groupby'] + [
                                                f"EAEL_{case}_timeseries_npv_{g}" for g in quantiles_list]]

                    risks = pd.merge(risks_ead,risks_eael,how='left',on=plot_types[st]['groupby'])

                    for g in quantiles_list:
                        risks[f'EAD_{case}_{g}'] = risks[f'EAD_{case}_timeseries_npv_{g}']/1e9
                        risks[f'EAEL_{case}_{g}'] = risks[f'EAEL_{case}_timeseries_npv_{g}']/1e9
                        risks[f'total_{case}_{g}'] = risks[f'EAD_{case}_{g}'] + risks[f'EAEL_{case}_{g}']

                    for rt in range(len(risk_types)): 
                        ax = ax_plots[j + len(sector_descriptions)*rt]  
                        y_ticks = []
                        for c in range(len(plot_types[st]['climate_scenarios'])):
                            sc = plot_types[st]['climate_scenarios'][c]
                            cl = plot_types[st]['scenario_color'][c]
                            m = plot_types[st]['scenario_marker'][c]
                            exp = risks[risks['rcp'] == sc]
                            ax.plot(exp['epoch'],
                                    exp[f"{risk_types[rt]}_{case}_{quantiles_list[0]}"],
                                    m,color=cl,markersize=4,linewidth=2.0,
                                    label=f"RCP {sc} mean")
                            ax.fill_between(exp['epoch'],exp[f"{risk_types[rt]}_{case}_{quantiles_list[0]}"],
                                exp[f"{risk_types[rt]}_{case}_{quantiles_list[1]}"],alpha=0.3,facecolor=cl)
                            ax.fill_between(exp['epoch'],exp[f"{risk_types[rt]}_{case}_{quantiles_list[0]}"],
                                exp[f"{risk_types[rt]}_{case}_{quantiles_list[2]}"],alpha=0.3,facecolor=cl,
                                label=f"RCP {sc} min-max")
                            if str(sc) == '4.5':
                                # y_ticks.append(
                                #                 exp[(
                                #                 exp[f"{risk_types[rt]}_{case}_mean"] > 0
                                #                 )][f"{risk_types[rt]}_{case}_mean"].min())
                                # y_ticks.append(
                                #                 exp[(
                                #                 exp[f"{risk_types[rt]}_{case}_mean"] > 0
                                #                 )][f"{risk_types[rt]}_{case}_mean"].max())
                                y_ticks.append(
                                                exp[(
                                                exp[f"{risk_types[rt]}_{case}_amin"] > 0
                                                )][f"{risk_types[rt]}_{case}_amin"].min())
                                y_ticks.append(
                                                exp[(
                                                exp[f"{risk_types[rt]}_{case}_amin"] > 0
                                                )][f"{risk_types[rt]}_{case}_amin"].max())
                            else:
                                y_ticks.append(
                                                exp[(
                                                exp[f"{risk_types[rt]}_{case}_mean"] > 0
                                                )][f"{risk_types[rt]}_{case}_mean"].max())
                                # y_ticks.append(
                                #                 exp[(
                                #                 exp[f"{risk_types[rt]}_{case}_amax"] > 0
                                #                 )][f"{risk_types[rt]}_{case}_amax"].min())
                                y_ticks.append(
                                                exp[(
                                                exp[f"{risk_types[rt]}_{case}_amax"] > 0
                                                )][f"{risk_types[rt]}_{case}_amax"].max())

                    
                        ax.set_xlabel('Timeline (year)',fontsize=14,fontweight='bold')
                        if case == 'avoided':
                            ax.set_ylabel(f'Avoided {risk_types_label[rt]} (RMB billions)',fontsize=14,fontweight='bold')
                        else:
                            ax.set_ylabel(f'{risk_types_label[rt]} (RMB billions)',fontsize=14,fontweight='bold')

                        # if j + len(sector_descriptions)*rt == 3:    
                        #     ax.legend(loc='upper left',bbox_to_anchor=(0.43,0.87),prop={'size':14,'weight':'bold'})
                        # ax.set_xscale('log')
                        # if sector['sector'] == 'road':
                        #     ax.set_yscale('log')
                        if axis_log_scale is True:
                            ax.set_yscale('log')
                        # ax.set_ylim(0,risks[f'total_{case}_max'].max())
                        ax.set_ylim(min(risks[f'EAD_{case}_amin'].min(),
                                        risks[f'EAEL_{case}_amin'].min()),
                                    1.1*risks[f'total_{case}_amax'].max())
                        ax.tick_params(axis='both', labelsize=14)

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
                        # print (y_ticks)
                        # ax.set_yscale('log')
                        ax.set_yticks(y_ticks)
                        ax.set_yticklabels([str(t) for t in y_ticks])
                        # ax.set_yscale('log')
                        ax.grid(True)
                        # ax.set_xticks([t for t in rps])
                        # ax.set_xticklabels([str(t) for t in rps])
                        # ax.set_xticks([t for t in list(set(exposures[exposures['year'] == baseyear]['return_period'].values))], 
                        #             [str(t) for t in list(set(exposures[exposures['year'] == baseyear]['return_period'].values))])
                        ax.text(
                            0.05,
                            0.95,
                            f"{figure_texts[j + len(sector_descriptions)*rt]}  Residual {risk_types_label[rt]}",
                            horizontalalignment='left',
                            transform=ax.transAxes,
                            size=16,
                            weight='bold')
                        # if risk_types[rt] == 'total':
                        #     ax.text(
                        #         0.35,
                        #         0.87,
                        #         f"Discounted {risk_types_label[rt]}",
                        #         horizontalalignment='left',
                        #         transform=ax.transAxes,
                        #         size=18,
                        #         weight='bold')
                        # else:
                        #     ax.text(
                        #         0.50,
                        #         0.87,
                        #         f"Discounted {risk_types_label[rt]}",
                        #         horizontalalignment='left',
                        #         transform=ax.transAxes,
                        #         size=18,
                        #         weight='bold')
                        if j + len(sector_descriptions)*rt <= 3:
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
                ax_plots[(len(risk_types) - 1)*len(sector_descriptions) - 1].legend(
                                        loc='upper left', 
                                        bbox_to_anchor=(1.05,0.8),
                                        prop={'size':18,'weight':'bold'})
                plt.tight_layout()
                if len(risk_types) == 2:
                    if axis_log_scale is True:
                        save_fig(os.path.join(figures,f"{plot_types[st]['plot_name']}_{case}_without_total_logscale.png"))
                    else:
                        save_fig(os.path.join(figures,f"{plot_types[st]['plot_name']}_{case}_without_total.png"))
                else:
                    if axis_log_scale is True:
                        save_fig(os.path.join(figures,f"{plot_types[st]['plot_name']}_{case}_logscale.png"))
                    else:
                        save_fig(os.path.join(figures,f"{plot_types[st]['plot_name']}_{case}.png"))
                plt.close()
    



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
