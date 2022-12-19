"""Generate hazard-damage curves
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import (MaxNLocator,LinearLocator, MultipleLocator)
import matplotlib.pyplot as plt
from matplotlib import cm
from map_plotting_utils import *
from tqdm import tqdm
tqdm.pandas()

mpl.style.use('ggplot')
mpl.rcParams['font.size'] = 10.
mpl.rcParams['font.family'] = 'tahoma'
mpl.rcParams['axes.labelsize'] = 12.
mpl.rcParams['xtick.labelsize'] = 12.
mpl.rcParams['ytick.labelsize'] = 12.

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

    return damage_percent

def main(config):
    incoming_path = config['paths']['incoming_data']
    base_path = config['paths']['data']
    figures_data_path = config['paths']['figures']

    sector_descriptions = [
                        {
                            'sector':'landfill',
                            'sector_name':'Landfill',
                            'sector_label':'Landfill sites',
                            'damage_source':'Kang et al. (2006)'
                            
                        },
                        {
                            'sector':'air',
                            'sector_name':'Airport',
                            'sector_label':'Airports',
                            'damage_source':'Kok et al. (2004)'
                            
                        },
                        {
                            'sector':'power',
                            'sector_name':'Power Plant',
                            'sector_label':'Power plants',
                            'damage_source':'FEMA (2013)'
                            
                        },
                        {
                            'sector':'road',
                            'sector_name':'Road',
                            'sector_label':'Roads',
                            'damage_source':'FEMA (2013)'
                        }
                        ]
    fig, ax_plots = plt.subplots(2,2,figsize=(12,12),dpi=500)
    ax_plots = ax_plots.flatten()
    for s in range(len(sector_descriptions)):
        sector = sector_descriptions[s]
        ax = ax_plots[s]
        data = pd.DataFrame()
        data['flood_depth'] = np.linspace(0,5,20)
        data["asset_type"] = sector["sector_name"]
        data['damage_percent_min'] = data.progress_apply(
                                        lambda x: assign_damage_percent(x,
                                                'flood_depth','asset_type',1),axis=1)
        data['damage_percent_max'] = data.progress_apply(
                                        lambda x: assign_damage_percent(x,
                                               'flood_depth','asset_type',5),axis=1)
        ax.plot(data['flood_depth'],
                data['damage_percent_min'],'-',
                color='#3182bd',linewidth=2.0,
                label=f"{sector['damage_source']} curve")    

        ax.fill_between(data['flood_depth'],
                        data['damage_percent_min'],
                        data['damage_percent_max'],
                        alpha=0.3,facecolor='#3182bd',
                        label='Uncertainty range')
        ax.legend(loc='lower right',prop={'size':14,'weight':'bold'})
        ax.set_xlabel('Flood depth (m)',fontweight='bold',fontsize=12)
        ax.set_ylabel('Damage percentage (%)',fontweight='bold',fontsize=12)

        ax.set_title(f"{sector['sector_label']}: Flood-depth vs damage",fontsize=14,fontweight='bold')
    plt.tight_layout()
    save_fig(os.path.join(figures_data_path,
                'damage_curves',    
                'damage_curves.png'))
    plt.close()    

    start_year = 2016
    end_year = 2080
    growth_rates = pd.read_excel(os.path.join(base_path,'growth_forecast',
                                'growth_forecast_OECD.xlsx'),
                                sheet_name='China_forecast').fillna(0)
    growth_year_rates = []
    growth_rates_times = list(sorted(growth_rates.TIME.values.tolist()))
    # And create parameter values
    for y in range(start_year,end_year+1):
        if y in growth_rates_times:
            growth_year_rates.append((y,growth_rates.loc[growth_rates.TIME == y,'Growth'].values[0]))
        elif y < growth_rates_times[0]:
            growth_year_rates.append((y,growth_rates.loc[growth_rates.TIME == growth_rates_times[0],'Growth'].values[0]))
        elif y > growth_rates_times[-1]:
            growth_year_rates.append((y,growth_rates.loc[growth_rates.TIME == growth_rates_times[-1],'Growth'].values[0]))

    growth_year_rates = pd.DataFrame(growth_year_rates,columns=['year','growth_rates'])
    fig, ax = plt.subplots(1,1,figsize=(6,6),dpi=500)
    ax.plot(growth_year_rates['year'],
                growth_year_rates['growth_rates'],'-',
                color='#de2d26',linewidth=2.0,
                marker='o',markersize=4,
                label=f"GDP growth forecast (OECD 2020)")    

    ax.fill_between(growth_year_rates['year'],
                    growth_year_rates['growth_rates'] - 2.0,
                    1.2*growth_year_rates['growth_rates'] + 2.0,
                    alpha=0.3,facecolor='#de2d26',
                    label='Uncertainty range')
    ax.legend(loc='upper right',prop={'size':14,'weight':'bold'})
    ax.set_xlabel('Year',fontweight='bold',fontsize=12)
    ax.set_ylabel('GDP growth rate (%)',fontweight='bold',fontsize=12)
    plt.tight_layout()
    save_fig(os.path.join(figures_data_path,
                'damage_curves',    
                'growth_rates.png'))
    plt.close()

    """Plot a sample for a loss-probability curve with residual and avoided risks  
    """
    data = pd.read_csv(os.path.join(
    					incoming_path,
    					"asset_details",
    					"landfill_intersection_cost_final.csv"
    					)
    				)
    cost_column = "closet_cost_info"
    data = data[data["year"] == 2016].groupby("probability")[cost_column].sum().reset_index()
    print (data)
    standard = 1.0/25
    fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=500)
    residual = data[data["probability"] <= standard]
    avoided = data[data["probability"] >= standard]

    ax.plot(residual["probability"],
                        residual[cost_column],'-',
                        marker='o',
                        color="#de2d26",linewidth=2.0,
                        label="Residual Damages (or Losses)")    

    ax.fill_between(residual["probability"],
                    np.array([0]*len(residual.index)),
                    residual[cost_column],
                    alpha=0.5,facecolor="#de2d26",
                    label="Residual Risk (EAD or EAL)")

    ax.plot(avoided["probability"],
                        avoided[cost_column],'-',
                        marker='o',
                        color="#08306b",linewidth=2.0,
                        label="Avoided Damages (or Losses)")    

    ax.fill_between(avoided["probability"],
                    np.array([0]*len(avoided.index)),
                    avoided[cost_column],
                    alpha=0.5,facecolor="#08306b",
                    label="Avoided Risk (EAD or EAL)")

    standard_line = np.array([0.0] + data[cost_column].values.tolist())
    ax.plot(np.array([standard]*len(standard_line)),standard_line,'-',color="#000000",linewidth=4.0,
                        label=f"{int(1.0/standard)}-year flood protection standard")

    ax.legend(loc='upper right',fontsize=14)
    ax.set_xlabel("Exceedance probability",fontweight='bold',fontsize=16)
    ax.set_ylabel('Damages or Losses (RMB millions)',fontweight='bold',fontsize=16)

    plt.tight_layout()
    save_fig(os.path.join(figures_data_path,"damage_curves",
            "example_loss_probability_curve.png"))
    plt.close()

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)

    