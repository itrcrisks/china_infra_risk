"""Risk plots
"""
import os
import sys
from collections import OrderedDict
import ast
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from map_plotting_utils import (load_config,get_projection, plot_basemap, 
                        plot_point_assets,point_map_plotting_color_width,
                        plot_line_assets,line_map_plotting_colors_width,
                        save_fig,line_map_plotting,point_map_plotting)
import seaborn as sns
from tqdm import tqdm
tqdm.pandas()


def confidence_interval_estimation(x_vals,y_vals,x_fit,CI=0,add=True):
    x_vals = np.where(x_vals < 1e-5,1e-5,x_vals)
    x = np.log(x_vals)
    # x = x_vals
    max_y = max(y_vals)
    y = y_vals/max(y_vals)
    y = np.log(y)

    n = y.size
    p, pcov = np.polyfit(x, y, deg=1, cov=True)
    yfit = np.polyval(p, x)                           # evaluate the polynomial at x
    perr = np.sqrt(np.diag(pcov))     # standard-deviation estimates for each coefficient
    R2 = np.corrcoef(x, y)[0, 1]**2  # coefficient of determination between x and y
    resid = y - yfit
    s_err = np.sqrt(np.sum(resid**2)/(n - 2))  # standard deviation of the error (residuals)
    
    # xfit = np.array([np.log(v) for v in np.arange(x_vals[0],x_vals[-1]+1)])
    xfit = np.array([np.log(v) for v in x_fit])
    yfit = np.polyval(p, xfit) 
    # print (yfit)
    yfit = max_y*np.exp(yfit)
    if CI > 0:
        # Confidence interval for the linear fit:
        t = stats.t.ppf(CI, n - 2)
        ci = t * s_err * np.sqrt(    1/n + (xfit - np.mean(xfit))**2/np.sum((xfit-np.mean(xfit))**2))
        ci = max_y*ci
    else:
        ci = np.array([0]*len(yfit))

    if add is True:
        yfit += ci
    else:
        yfit = yfit - ci

    yfit = np.where(yfit < 0,0,yfit)

    return yfit

def get_risk_benefit_cost_curves(vals,CI=0,add=True):

    protection_levels = np.array([v[0] for v in vals])
    risks = np.array([v[1] for v in vals])
    undefended_risks = [v[2] for v in vals][0]
    costs = np.array([v[3] for v in vals])

    # protection_fit = np.arange(protection_levels[0],protection_levels[-1]+1)
    protection_fit = np.arange(protection_levels[0],10001)
    risk_fit = confidence_interval_estimation(protection_levels,risks,protection_fit,CI=CI,add=add)
    benefit_fit = undefended_risks - risk_fit
    benefit_fit = np.where(benefit_fit < 0,0,benefit_fit)
    cost_fit = confidence_interval_estimation(protection_levels,costs,protection_fit,CI=CI,add=add)

    return list(zip(protection_fit,risk_fit,benefit_fit,cost_fit))

def main(config):
    data_path = config['paths']['data']
    output_path = config['paths']['output']
    figure_path = config['paths']['figures']
    network_path = os.path.join(data_path,'network') # Where we have all the network shapefiles
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
                            'length_unit':None, # To convert length in km to meters
                            'sector_marker':'o',
                            'sector_size':12.0,
                            'sector_color':'#74c476',
                            'exposure_column':'assets'
                        },
                        {
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
                            'length_unit':None, # To convert length in km to meters
                            'sector_marker':'s',
                            'sector_size':12.0,
                            'sector_color':'#636363',
                            'exposure_column':'assets'
                        },
                        {
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
                            'length_unit':None, # To convert length in km to meters
                            'sector_marker':'^',
                            'sector_size':12.0,
                            'sector_color':'#fb6a4a',
                            'exposure_column':'assets'
                        },
                        {
                            'sector':'road',
                            'sector_name':'Road',
                            'sector_label':'Roads',
                            'asset_type':'edges',
                            'sector_shapefile':'final_road.shp',
                            'id_column':'road_ID',
                            'adaptation_criteria_column':'grade',
                            'flood_protection_column':'flood_pro',
                            'flood_protection_column_assets':'flood_pro',
                            'cost_column':'best_cost_per_km_sec', # Asset value in 100 million RMB/km
                            'cost_conversion':1.0e8, # Convert Asset value to RMB
                            'min_economic_loss_column':['loss_income_min'], # Daily Economic loss RMB
                            'max_economic_loss_column':['loss_income_max'], # Daily Economic loss RMB
                            'economic_loss_conversion': 1.0, # Convert Annual Losses to Daily in RMB
                            'length_column':'road_length_km',
                            'length_unit':1000.0, # To convert length in km to meters
                            'sector_marker':None,
                            'sector_size':1.0,
                            'sector_color':'#969696',
                            'exposure_column':'length'
                        }
                        ]
    baseyear = 2016
    flood_color = '#3182bd'
    noflood_color = '#969696'
    flood_colors = ['#9ecae1','#6baed6','#2171b5','#08306b']
    [2,5,10,25,50,100,250,500,1000]
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
    benefit_type = 'median'
    risk_type = ['direct','total']
    risk_type = 'total'
    sum_cols = ['_EAD_','_EAEL_','_cost_']
    # """
    # Step 1: Get all the Administrative boundary files
    #     Combine the English provinces names with the Chinese ones  
    # """
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

    # """
    # Step 2: Get map bounds to set the map projections  
    # """
    # bounds = boundary_gpd.geometry.total_bounds # this gives your boundaries of the map as (xmin,ymin,xmax,ymax)
    # ax_proj = get_projection(extent = (bounds[0]+5,bounds[2]-10,bounds[1],bounds[3]))
    # del admin_gpd

    """Get all the exposure plots
    """
    fig, ax_plots = plt.subplots(1,len(sector_descriptions),
                            figsize=(30,16),
                            dpi=500)
    ax_plots = ax_plots.flatten()
    j = 0
    for sector in sector_descriptions:
        sector_data = pd.read_csv(os.path.join(output_path,
                                'investment_timeseries',
                                f"{sector['sector']}_investment_timeseries_climate_scenarios_optimals_asset_level.csv")
        
                                )
        # sector_data = sector_data[sector_data["optimal_bcr_median_direct_mean_fit"]>= 1]
        # sector_data["bcr_one"] = abs(sector_data["optimal_bcr_median_direct_mean_fit"] - 1)
        # sector_data = sector_data.sort_values(by="bcr_one",ascending=True)
        sector_data = sector_data.sort_values(by="optimal_bcr_median_direct_mean_fit",ascending=False)
        # print (sector_data["bcr_one"].values[0],sector_data["optimal_bcr_median_direct_mean_fit"].values[0])
        # sector_data = sector_data.sort_values(by="optimal_bcr_median_total_mean_fit",ascending=False)
        obs = ast.literal_eval(sector_data['flood_protection_risks_benefits_costs_median_direct'].values[0])
        vals = get_risk_benefit_cost_curves(obs)
        # print (obs)
        protection_levels = np.array([v[0] for v in vals])
        # risks = np.array([v[2] for v in vals])
        # costs = np.array([v[3] for v in vals])
        bcr = np.array([v[2]/v[3] for v in vals])


        # print (sector_data['optimal_flood_protection_marginal_based'].values[0])
        # print (sector_data['optimal_flood_protection_risks_benefits_costs_bcr'].values[0])
        ax = ax_plots[j]
        # ax.plot(protection_levels, risks, color='k')
        ax.plot(protection_levels, bcr, color='k')
        # ax.scatter(obs_protection,obs_risks,marker='o',
        #                             s=30,
        #                             color='#1a1a1a',zorder=20)
        obs = ast.literal_eval(sector_data['flood_protection_risks_benefits_costs_median_total'].values[0])
        vals = get_risk_benefit_cost_curves(obs)
        # print (obs)
        protection_levels = np.array([v[0] for v in vals])
        # risks = np.array([v[2] for v in vals])
        # costs = np.array([v[3] for v in vals])
        bcr = np.array([v[2]/v[3] for v in vals])
        ax.plot(protection_levels, bcr, color='g')

        ax.set_xlabel('Protection level (years)')
        ax.set_ylabel('Marginal Risks and Costs (RMB)')

        # ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        # ax1.plot(protection_levels,costs, color='g')
        # ax.plot(protection_levels,costs, color='g')
        # ax1.scatter(obs_protection,obs_costs,marker='s',
        #                             s=30,
        #                             color='#2171b5',zorder=20)
        # ax1.set_ylabel('Marginal Costs')  # we already handled the x-label with ax1

        ax.set_xscale('log')
        ax.grid(True)
        j += 1

    plt.tight_layout()
    save_fig(os.path.join(figures,
            f"asset_investment_marginal_benefits_costs.png"))
    
    plt.close()

    # sector = sector_descriptions[0]
    # sector_df = pd.read_excel(os.path.join(output_path,
    #                             'investment_timeseries',
    #                             'investment_timeseries_climate_scenarios_optimals_global.xlsx'),
    #                             sheet_name='landfill')
    # # print (sector_df)


    # # # vals = sorted(sector_df['optimal_flood_protection_benefit_cost_marginal'].values[0],key = lambda x:x[0])
    # vals = sector_df['flood_protection_risks_benefits_costs'].values[0]
    # vals = ast.literal_eval(vals)
    # # print (vals)

    # # # print (sector_df['optimal_flood_protection_benefit_cost_marginal'].values[0])

    # protection_levels = np.array([v[0] for v in vals])
    # # print (protection_levels)
    # risks = np.array([v[1] for v in vals])
    # benefits = np.array([v[2] for v in vals])
    # costs = np.array([v[3] for v in vals])

    # protection_fit = np.arange(protection_levels[0],protection_levels[-1]+1)
    # # protection_fit = protection_levels
    # # risk_fit_mean = fit_curves(protection_levels,risks,CI=0)
    # # risk_fit_q95 = fit_curves(protection_levels,risks,CI=2)
    # # risk_fit_q5 = fit_curves(protection_levels,risks,CI=-2)
    # risk_fit_mean, risk_ci = confidence_interval_estimation(protection_levels,risks)
    # # print (risk_fit_mean)

    # # print (risk_fit)
    # fig, ax1 = plt.subplots()
    # ax1.plot(protection_levels,risks,'o')
    # ax1.plot(protection_fit,risk_fit_mean,'-')
    # ax1.plot(protection_fit,risk_fit_mean + risk_ci,'.-')
    # ax1.plot(protection_fit,risk_fit_mean - risk_ci,'.-')
    # plt.show()

    # protection_fit = np.arange(protection_levels[0],protection_levels[-1]+1)
    # benefit_fit = fit_curves(protection_levels,benefits)
    # # print (risk_fit)
    # fig, ax1 = plt.subplots()
    # ax1.plot(protection_levels,benefits,'o')
    # ax1.plot(protection_fit,benefit_fit,'x-')
    # plt.show()

    # protection_fit = np.arange(protection_levels[0],protection_levels[-1]+1)
    # cost_fit = fit_curves(protection_levels,costs)
    # # print (risk_fit)
    # fig, ax1 = plt.subplots()
    # ax1.plot(protection_levels,costs,'o')
    # ax1.plot(protection_fit,cost_fit,'x-')
    # plt.show()

    # marginal_bcr = np.array([v[-1] for v in vals])
    # # print (risks)
    # # print (min(costs),max(costs),min(costs)/max(costs))
    # # risks = risks/max(risks)
    # # costs = costs/max(costs)
    # # risk_fit = []
    # # cost_fit = []
    # # protection_fit = []
    # # for p in range(len(protection_levels)-1):
    # #     risk_popt, risk_pcov = curve_fit(func_exp, 
    # #                                 np.array([protection_levels[p],protection_levels[p+1]]), 
    # #                                 np.array([risks[p],risks[p+1]]))
    # #     cost_popt, cost_pcov = curve_fit(func_exp, 
    # #                                 np.array([protection_levels[p],protection_levels[p+1]]), 
    # #                                 np.array([costs[p],costs[p+1]]))

    # #     protection_fit += list(np.linspace(protection_levels[p],protection_levels[p+1],20))
    # #     risk_fit += list(func_exp(np.linspace(protection_levels[p],protection_levels[p+1],20),*risk_popt))
    # #     cost_fit += list(func_exp(np.linspace(protection_levels[p],protection_levels[p+1],20),*cost_popt))


    # risk_popt, risk_pcov = curve_fit(func_log_inverse, protection_levels, risks)
    # cost_popt, cost_pcov = curve_fit(func_log_inverse, protection_levels, costs)
    # print (risk_popt)
    # print (cost_popt)
    # # print (np.linspace(protection_levels[0],protection_levels[-1],20))
    # protection_fit = np.arange(protection_levels[0],protection_levels[-1]+1)
    # risk_fit = func_log_inverse(protection_fit,*risk_popt)
    # cost_fit = func_log_inverse(protection_fit,*cost_popt)

    # print ((risk_popt[2] - cost_popt[2])/(risk_popt[0]*risk_popt[1] - cost_popt[0]*cost_popt[1]))

    # risk_fit[risk_fit <0] = 0
    # cost_fit[cost_fit <0] = 0
    # vals = list(zip(protection_fit,risk_fit,cost_fit))
    # print (sorted(vals,key = lambda v:v[1]/v[2], reverse=True)[:10])

    # vals = list(zip(protection_fit[1:],np.diff(risk_fit),np.diff(cost_fit)))
    # print (sorted(vals,key = lambda v:v[1]/v[2], reverse=False)[:10])

    # # # # print (*risk_popt)
    # # # # print (risk_fit)
    # # # # print (cost_fit)
    # # # # pr_levels = np.linspace(protection_levels[0],protection_levels[-1],20)
    # # # marginals = []
    # # # for c in range(len(protection_fit)-1):
    # # #     marginal_benefit = risk_fit[c] - risk_fit[c+1]
    # # #     marginal_cost = cost_fit[c+1] - cost_fit[c]
    # # #     if marginal_cost > 0:
    # # #         marginals.append((int(protection_fit[c]),int(protection_fit[c+1]),marginal_benefit,marginal_cost,marginal_benefit/marginal_cost))

    # # # if len(marginals) > 0:
    # # #     return sorted(marginals,key = lambda v:v[-1], reverse=True)
    # # # else:
    # # #     return [(x[sector['flood_protection_column']],0,0,0)]

    # # # print (sorted(marginals,key = lambda v:v[-1], reverse=True))


    # fig, ax1 = plt.subplots()

    # color = 'tab:red'
    # ax1.set_xlabel('Protection level (years)')
    # ax1.set_ylabel('Risks', color=color)
    # # ax1.plot(protection_levels, risks, color=color)
    # # ax1.plot(protection_levels, risks+costs, color='k')
    # # ax1.plot(protection_levels, marginal_bcr, color=color)
    # ax1.plot(np.array(protection_fit), np.array(risk_fit), color='k')
    # # ax1.plot(np.array(protection_fit[1:]), np.array([abs(d) for d in np.diff(risk_fit)]), color='k')

    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('Costs', color=color)  # we already handled the x-label with ax1
    # # ax2.plot(protection_levels, costs, color=color)
    # ax2.plot(np.array(protection_fit), np.array(cost_fit), color='g')
    # # ax2.plot(np.array(protection_fit[1:]),np.diff(cost_fit),color='g')
    # ax2.tick_params(axis='y', labelcolor=color)

    # # print ([abs(d) for d in np.diff(risk_fit)])
    # # print (np.diff(cost_fit))

    # # idx = np.argwhere(min(risks + costs)).flatten()
    # # print (idx)
    # print (min(np.diff(risks)/np.diff(costs)))
    # a = np.diff(risk_fit)/np.diff(cost_fit)
    # print (a[:10])
    # idx = np.argwhere(a == min(a)).flatten()
    # # idx = np.argwhere(np.diff(risk_fit)/np.diff(cost_fit) == 1).flatten()
    # print (idx)
    # ax1.plot(protection_fit[idx-1], risk_fit[idx-1], 'ro')
    # # ax1.plot(protection_fit[1:],np.diff(risk_fit)/np.diff(cost_fit))

    # # ax2.plot(protection_levels[idx], costs[idx], 'ro')
    # # ax2.plot(protection_levels, risks+costs, color='k')

    # # sector_df['optimal_flood_protection_benefit_cost_marginal'] = sector_df.progress_apply(lambda x: get_optimal_protection_marginal(
    # #                                                             x,protection_columns,sector,benefit_type,risk_type='total'),axis=1)
    # # # print (sector_df['optimal_flood_protection_benefit_cost_marginal'].values[0])

    # # # vals = sorted(sector_df['optimal_flood_protection_benefit_cost_marginal'].values[0],key = lambda x:x[0])
    # # vals = sector_df['optimal_flood_protection_benefit_cost_marginal'].values[0]
    # # print (vals)

    # # # print (sector_df['optimal_flood_protection_benefit_cost_marginal'].values[0])

    # # protection_levels = np.array([v[0] for v in vals])
    # # risks = 1e-6*np.array([v[1] for v in vals])
    # # costs = 1e-6*np.array([v[-1] for v in vals])
    # # ax1.plot(protection_levels, risks, color=color)


    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.savefig('test2.png')
    # plt.close()
    # # plt.show()




if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
