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
import risk_and_adaptation_functions as rad
from tqdm import tqdm
tqdm.pandas()

def confidence_interval_estimation(x_vals,y_vals,x_fit,CI=0,add=True):
    # print (x_vals,y_vals)
    x_vals = np.where(x_vals < 1e-5,1e-5,x_vals)
    x = np.log(x_vals)
    # x = x_vals
    max_y = max(y_vals)
    y = y_vals/max(y_vals)
    y = np.where(y < 1e-5,1e-5,y)
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

def get_risk_benefit_cost_curves_old(vals,CI=0,add=True):

    protection_levels = np.array([v[0] for v in vals])
    risks = np.array([v[1] for v in vals])
    benefits = np.array([v[2] for v in vals])
    costs = np.array([v[3] for v in vals])

    # protection_fit = np.arange(protection_levels[0],protection_levels[-1]+1)
    protection_fit = np.arange(protection_levels[0],10001)
    risk_fit = confidence_interval_estimation(protection_levels,risks,protection_fit,CI=CI,add=add)
    benefit_fit = confidence_interval_estimation(protection_levels,benefits,protection_fit,CI=CI,add=add)
    cost_fit = confidence_interval_estimation(protection_levels,costs,protection_fit,CI=CI,add=add)

    return list(zip(protection_fit,risk_fit,benefit_fit,cost_fit))

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

def get_marginal_risk_benefit_cost_curves(vals,CI=0,add=True):

    protection_levels = np.array([v[0] for v in vals])
    risks = np.array([v[1] for v in vals])
    costs = np.array([v[2] for v in vals])

    # protection_fit = np.arange(protection_levels[0],protection_levels[-1]+1)
    protection_fit = np.arange(protection_levels[0],10001)
    benefit_fit = confidence_interval_estimation(protection_levels,risks,protection_fit,CI=CI,add=add)
    # benefit_fit = undefended_risks - risk_fit
    benefit_fit = np.where(benefit_fit < 0,0,benefit_fit)
    cost_fit = confidence_interval_estimation(protection_levels,costs,protection_fit,CI=CI,add=add)

    return list(zip(protection_fit,benefit_fit,cost_fit))

def get_max_benefit_cost_ratios(vals):
    bcr_list = []
    for v in vals:
        # print (v)
        if v[-1] == 0:
            bcr_list.append(tuple(list(v)+[0]))
        else:
            bcr_list.append(tuple(list(v)+[v[-2]/v[-1]]))

    return sorted(bcr_list,key = lambda v:v[-1], reverse=True)[0]

def get_marginal_risks_costs(vals):
    marginal_list = []
    initial_design = vals[0]
    marginal_list.append((initial_design[0],
    					initial_design[2]-initial_design[1],
    					initial_design[3]))   
    for v in range(len(vals)-1):
        # print (v)
        upgrade_st = vals[v+1][0]
        mc_risk = vals[v][1] - vals[v+1][1]
        mc_cost = vals[v+1][-1] - vals[v][-1]
        # mc_diff = mc_risk - mc_cost
        marginal_list.append((upgrade_st,mc_risk,mc_cost))

    return marginal_list

def get_optimal_protection_marginal(vals):
    marginal_vals = get_marginal_risks_costs(vals)
    marginal_vals = [v for v in marginal_vals[1:] if v[-1] >= 0]
    if marginal_vals:
        marginal_optimal = sorted(marginal_vals,key = lambda v:v[-1], reverse=False)[0]
        optimal_st = marginal_optimal[0]
        return get_max_benefit_cost_ratios([v for v in vals if v[0] == optimal_st]) 
    else:
        return get_max_benefit_cost_ratios([vals[0]])  


# def get_protection_benefits_costs(x,protection_columns,sector,
#                             benefit_column_type,
#                             risk_type='total'):
#     cba = []
#     for pc in protection_columns:
#         if pc == sector['flood_protection_column']:
#             ead_col = f'total_EAD_designed_protection_npv_{benefit_column_type}'
#             eael_col = f'total_EAEL_designed_protection_npv_{benefit_column_type}'
#             avoided_ead_col = f'total_EAD_designed_protection_avoided_npv_{benefit_column_type}'
#             avoided_eael_col = f'total_EAEL_designed_protection_avoided_npv_{benefit_column_type}' 
#             cost_col = 'median_total_adapt_cost_npv_designed_protection'
#         else:
#             ead_col = f'total_EAD_{pc}_npv_{benefit_column_type}'
#             eael_col = f'total_EAEL_{pc}_npv_{benefit_column_type}'
#             avoided_ead_col = f'total_EAD_{pc}_avoided_npv_{benefit_column_type}'
#             avoided_eael_col = f'total_EAEL_{pc}_avoided_npv_{benefit_column_type}'
#             cost_col = f'median_total_adapt_cost_npv_{pc}'

#         if risk_type == 'direct':
#             cba.append((x[pc],x[ead_col],x[avoided_ead_col],x[cost_col]))
#         elif risk_type == 'indirect':
#             cba.append((x[pc],x[eael_col],x[avoided_eael_col],x[cost_col]))
#         else:
#             cba.append((x[pc],x[ead_col] + x[eael_col],x[avoided_ead_col] + x[avoided_eael_col],x[cost_col]))

#     cba = [c for c in cba if str(c[0]) != 'nan']
#     cba = sorted(list(set(cba)),key = lambda v:v[0], reverse=False)
#     return cba

def get_protection_risks_costs(x,protection_columns,
							flood_protection_column,
							no_protection_column,
                            benefit_column_type,
                            risk_type='total'):
    cba = []
    for pc in protection_columns:
        undefended_ead_col = f'EAD_river_{no_protection_column}_npv_{benefit_column_type}'
        undefended_eael_col = f'EAEL_river_{no_protection_column}_npv_{benefit_column_type}' 
        if pc == flood_protection_column:
            ead_col = f'EAD_river_{flood_protection_column}_npv_{benefit_column_type}'
            eael_col = f'EAEL_river_{flood_protection_column}_npv_{benefit_column_type}'
            cost_col = f'mean_total_adapt_cost_npv_{flood_protection_column}_{benefit_column_type}'
            # cost_col = f'mean_ini_adapt_cost_{flood_protection_column}_{flood_protection_column}'
        else:
            ead_col = f'EAD_river_{pc}_npv_{benefit_column_type}'
            eael_col = f'EAEL_river_{pc}_npv_{benefit_column_type}'
            cost_col = f'mean_total_adapt_cost_npv_{pc}_{benefit_column_type}'
            # cost_col = f'mean_ini_adapt_cost_{pc}_{pc}'

        if risk_type == 'direct':
            cba.append((x[pc],x[ead_col],x[undefended_ead_col],x[cost_col]))
        elif risk_type == 'indirect':
            cba.append((x[pc],x[eael_col],x[undefended_eael_col],x[cost_col]))
        else:
            cba.append((x[pc],x[ead_col] + x[eael_col],x[undefended_ead_col] + x[undefended_eael_col],x[cost_col]))

    cba = [c for c in cba if str(c[0]) != 'nan']
    cba = sorted(list(set(cba)),key = lambda v:v[0], reverse=False)
    return cba

# def get_marginal_risk_benefit_costs_bcr_values(sector_df,protection_columns,
# 									flood_protection_column,
# 									no_protection_column,
# 									benefit_type,risk_type,CI,CI_label,asset_level=True):    
#     sector_df[f'flood_protection_risks_benefits_costs_{benefit_type}_{risk_type}'] = sector_df.progress_apply(
#                                                             lambda x: get_protection_risks_costs(
#                                                             x,protection_columns,
#                                                             flood_protection_column,
#                                                             no_protection_column,benefit_type,risk_type=risk_type),axis=1)

#     # sector_df['flood_protection_risks_benefits_costs_fits'] = sector_df.progress_apply(lambda x: get_risk_benefit_cost_curves(
#     #                                                         x['flood_protection_risks_benefits_costs'],CI=CI),axis=1)
#     # print (sector_df)
#     sector_df[f'flood_protection_risks_costs_marginals_{benefit_type}_{risk_type}'] = sector_df.progress_apply(
#     												lambda x: get_marginal_risks_costs(
#                                                     x[f'flood_protection_risks_benefits_costs_{benefit_type}_{risk_type}']),
#                                                     axis=1)

#     sector_df['flood_protection_risks_benefits_costs_fits'] = sector_df.progress_apply(lambda x: get_marginal_risk_benefit_cost_curves(
#                                                             x[f'flood_protection_risks_costs_marginals_{benefit_type}_{risk_type}'],
#                                                             CI=CI),axis=1)

#     # print(sector_df['flood_protection_risks_benefits_costs_fits'])
#     sector_df['optimal_flood_protection_risks_benefits_costs_bcr'] = sector_df.progress_apply(lambda x: get_max_benefit_cost_ratios(
#                                                                     x['flood_protection_risks_benefits_costs_fits']),axis=1)

#     # print (sector_df['optimal_flood_protection_benefit_cost'])
#     sector_df[[f'optimal_flood_protection_{benefit_type}_{risk_type}_{CI_label}',
#                f'optimal_benefit_{benefit_type}_{risk_type}_{CI_label}',
#                f'optimal_cost_{benefit_type}_{risk_type}_{CI_label}',
#                f'optimal_bcr_{benefit_type}_{risk_type}_{CI_label}']] = sector_df['optimal_flood_protection_risks_benefits_costs_bcr'].apply(pd.Series)

#     risk_columns = [f'flood_protection_risks_costs_marginals_{benefit_type}_{risk_type}',
#                     f'optimal_flood_protection_{benefit_type}_{risk_type}_{CI_label}',
#                     f'optimal_benefit_{benefit_type}_{risk_type}_{CI_label}',
#                     f'optimal_cost_{benefit_type}_{risk_type}_{CI_label}',
#                     f'optimal_bcr_{benefit_type}_{risk_type}_{CI_label}'] 

#     sector_df.drop(['flood_protection_risks_benefits_costs_fits',
#                     'optimal_flood_protection_risks_benefits_costs_bcr'],
#                     axis=1,
#                     inplace=True)           


#     return sector_df, risk_columns
    # print(sector_df['flood_protection_risks_benefits_costs_fits'])
    # sector_df['optimal_flood_protection_risks_benefits_costs_bcr'] = sector_df.progress_apply(lambda x: get_max_benefit_cost_ratios(
    #                                                                 x['flood_protection_risks_benefits_costs_fits']),axis=1)

    # # print (sector_df['optimal_flood_protection_benefit_cost'])
    # sector_df[[f'optimal_flood_protection_{benefit_type}_{risk_type}_{CI_label}',
    #            f'optimal_risk_{benefit_type}_{risk_type}_{CI_label}',
    #            f'optimal_benefit_{benefit_type}_{risk_type}_{CI_label}',
    #            f'optimal_cost_{benefit_type}_{risk_type}_{CI_label}',
    #            f'optimal_bcr_{benefit_type}_{risk_type}_{CI_label}']] = sector_df['optimal_flood_protection_risks_benefits_costs_bcr'].apply(pd.Series)

    # sector_df['optimal_flood_protection_marginal_based'] = sector_df.progress_apply(lambda x: get_optimal_protection_marginal(
    #                                                                 x['flood_protection_risks_benefits_costs_fits']),axis=1)

    # sector_df[[f'marginal_optimal_flood_protection_{benefit_type}_{risk_type}_{CI_label}',
    #            f'marginal_optimal_risk_{benefit_type}_{risk_type}_{CI_label}',
    #            f'marginal_optimal_benefit_{benefit_type}_{risk_type}_{CI_label}',
    #            f'marginal_optimal_cost_{benefit_type}_{risk_type}_{CI_label}',
    #            f'marginal_optimal_bcr_{benefit_type}_{risk_type}_{CI_label}'
    #            ]] = sector_df['optimal_flood_protection_marginal_based'].apply(pd.Series)


    # return sector_df

def get_risk_benefit_costs_bcr_values(sector_df,protection_columns,
										flood_protection_column,
										no_protection_column,
										benefit_type,risk_type,CI,CI_label,asset_level=True):   
    sector_df[f'flood_protection_risks_benefits_costs_{benefit_type}_{risk_type}'] = sector_df.progress_apply(
                                                            lambda x: get_protection_risks_costs(
                                                            x,protection_columns,
                                                            flood_protection_column,
                                                            no_protection_column,benefit_type,risk_type=risk_type),axis=1)
    sector_df['flood_protection_risks_benefits_costs_fits'] = sector_df.progress_apply(lambda x: get_risk_benefit_cost_curves(
                                                            x[f'flood_protection_risks_benefits_costs_{benefit_type}_{risk_type}'],
                                                            CI=CI),axis=1)

    # print(sector_df['flood_protection_risks_benefits_costs_fits'])
    sector_df['optimal_flood_protection_risks_benefits_costs_bcr'] = sector_df.progress_apply(lambda x: get_max_benefit_cost_ratios(
                                                                    x['flood_protection_risks_benefits_costs_fits']),axis=1)

    # print (sector_df['optimal_flood_protection_benefit_cost'])
    sector_df[[f'optimal_flood_protection_{benefit_type}_{risk_type}_{CI_label}',
               f'optimal_risk_{benefit_type}_{risk_type}_{CI_label}',
               f'optimal_benefit_{benefit_type}_{risk_type}_{CI_label}',
               f'optimal_cost_{benefit_type}_{risk_type}_{CI_label}',
               f'optimal_bcr_{benefit_type}_{risk_type}_{CI_label}']] = sector_df['optimal_flood_protection_risks_benefits_costs_bcr'].apply(pd.Series)

    risk_columns = [f'flood_protection_risks_benefits_costs_{benefit_type}_{risk_type}',
                    f'optimal_flood_protection_{benefit_type}_{risk_type}_{CI_label}',
                    f'optimal_risk_{benefit_type}_{risk_type}_{CI_label}',
                    f'optimal_benefit_{benefit_type}_{risk_type}_{CI_label}',
                    f'optimal_cost_{benefit_type}_{risk_type}_{CI_label}',
                    f'optimal_bcr_{benefit_type}_{risk_type}_{CI_label}'] 

    sector_df.drop(['flood_protection_risks_benefits_costs_fits',
                    'optimal_flood_protection_risks_benefits_costs_bcr'],
                    axis=1,
                    inplace=True)           


    return sector_df, risk_columns


def main(config):
    processed_data_path = config['paths']['data']
    output_path = config['paths']['output']
    investment_data = os.path.join(output_path,"benefits_costs_modified")
    investment_types = [
                    # {
                    # 'type':'sector_protection_levels',
                    # 'timeseries_filename':'benefits_costs_climate_scenarios_optimals_global_max.csv',
                    # 'optimal_values_filename':'investment_optimal_climate_scenarios_optimals_global_max.csv',
                    # 'asset_level':False,
                    # 'groupby':[
                    #             'climate_scenario',
                    #         ]
                    # },
                    {
                    'type':'asset_protection_levels',
                    'timeseries_filename':'benefits_costs_climate_scenarios_optimals_asset_level',
                    'optimal_values_filename':'investment_optimal_climate_scenarios_optimals_asset_level',
                    'asset_level':True,
                    'groupby':[]
                    },

                ]
    division_factor = 1e6
    benefit_type = 'amin'
    risk_types = ['direct','total']
    # risk_type = 'total'
    sum_cols = ['_EAD_','_EAEL_','_cost_']

    timeseries_columns = ['flood_protection_risks_benefits_costs',
                            'flood_protection_risks_benefits_costs_fits',
                            'flood_protection_risks_costs_marginals'
                        ]
    """Get all the exposure plots
    """
    # length_division = 1000.0 # Convert m to km
    network_csv = os.path.join(processed_data_path,
                            "network_layers_hazard_intersections_details.csv")
    sector_descriptions = pd.read_csv(network_csv)
    no_protection_column = "no_protection_rp"
    for pt in investment_types:
        for asset_info in sector_descriptions.itertuples():
            asset_sector = asset_info.asset_gpkg
            asset_id = asset_info.asset_id_column
            flood_protection_column = asset_info.flood_protection_column
            sector_data = pd.read_excel(os.path.join(investment_data,
                                    f"{asset_sector}_benefits_costs_climate_scenarios.xlsx"),
                                    sheet_name=asset_sector)
            sector_data = sector_data[sector_data["EAD_river_no_protection_rp_npv_amax"]>0]
            pr_cols = list(set([col for col in sector_data.columns.values.tolist() if '_to_' in col]))
            protection_columns = []
            for col in pr_cols: 
                fn = col.split('_to_')
                protection_columns.append(f"{fn[0].split('_')[-1]}_to_{fn[1].split('_')[0]}_year_protection")

            cols = []
            for sc in sum_cols:
                cols += [c for c in sector_data.columns.values.tolist() if sc in c]
            risk_cols = []
            for sc in sum_cols[:-1]:
                risk_cols += [c for c in sector_data.columns.values.tolist() if sc in c]    

            cost_cols = [c for c in sector_data.columns.values.tolist() if '_cost_' in c] 

            sector_data[cols][sector_data[cols] <= 1e-9] = 0
            sector_data = sector_data[(sector_data[risk_cols].sum(axis=1) > 0) | (sector_data[cost_cols].sum(axis=1) > 0)]


            
            protection_columns = [flood_protection_column] + list(set(protection_columns))

            if pt['asset_level'] is False:
                sector_df = sector_data.groupby(['rcp'] + protection_columns)[cols].sum().reset_index()
            else:
                sector_df = sector_data.copy()

            sector_columns = []
            for risk_type in risk_types:
                sector_df, sector_cols = get_risk_benefit_costs_bcr_values(sector_df,
                                                    protection_columns,
                                                    flood_protection_column,
                                                    no_protection_column,
                                                    benefit_type,risk_type,0,'mean_fit',asset_level=pt['asset_level'])
                sector_columns += sector_cols
            
            sector_df[[asset_id,"rcp"] + sector_columns].to_csv(os.path.join(
                                                            investment_data,
                                                            f"{asset_sector}_{pt['timeseries_filename']}_{benefit_type}.csv"
                                                            ),
                                                            index=False)

            # for risk_type in risk_types:
            #     sector_df, sector_cols = get_marginal_risk_benefit_costs_bcr_values(sector_df,
            #                                         protection_columns,
            #                                         flood_protection_column,
            #                                         no_protection_column,
            #                                         benefit_type,risk_type,0,'mean_fit',asset_level=pt['asset_level'])
            #     sector_columns += sector_cols
            
            # sector_df[[asset_id,"rcp"] + sector_columns].to_csv(os.path.join(
            #                                                 investment_data,
            #                                                 f"{asset_sector}_marginal_{pt['timeseries_filename']}"
            #                                                 ),
            #                                                 index=False)
            # sector_df.drop(timeseries_columns,axis=1).to_csv(
            #                                                 os.path.join(investment_data,
            #                                                 f"{sector['sector']}_{pt['optimal_values_filename']}"
            #                                                 ),
            #                                                 index=False)
            # sector_df.to_excel(output_wrtr,sector['sector'], index=False)





if __name__ == '__main__':
    CONFIG = rad.load_config()
    main(CONFIG)
