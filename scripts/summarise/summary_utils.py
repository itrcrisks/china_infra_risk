"""Functions for preprocessing data
"""
import sys
import os
import json
from collections import OrderedDict
import pandas as pd
import geopandas as gpd
import zipfile
import numpy as np
import ast
from tqdm import tqdm
tqdm.pandas()

def load_config():
    """Read config.json
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    return config


def grouping_values(dataframe,grouping_by_columns,grouped_columns,add_values=True):
    if add_values is False:
        df = dataframe.groupby(grouping_by_columns
                            )[grouped_columns].agg(lambda x:len(x.unique())).reset_index(name='assets')

    else:
        df = dataframe.groupby(grouping_by_columns
                            )[grouped_columns].sum().reset_index()

    return df

# def quantiles(dataframe,grouping_by_columns,grouped_columns):
#     quantiles_list = ['mean','min','max','median','q5','q95']
#     df_list = []
#     for quant in quantiles_list:
#         if quant == 'mean':
#             # print (dataframe)
#             df = dataframe.groupby(grouping_by_columns)[grouped_columns].mean()
#         elif quant == 'min':
#             df = dataframe.groupby(grouping_by_columns)[grouped_columns].min()
#         elif quant == 'max':
#             df = dataframe.groupby(grouping_by_columns)[grouped_columns].max()
#         elif quant == 'median':
#             df = dataframe.groupby(grouping_by_columns)[grouped_columns].quantile(0.5)
#         elif quant == 'q5':
#             df = dataframe.groupby(grouping_by_columns)[grouped_columns].quantile(0.05)
#         elif quant == 'q95':
#             df = dataframe.groupby(grouping_by_columns)[grouped_columns].quantile(0.95)

#         df.rename(columns=dict((g,f"{g}_{quant}") for g in grouped_columns),inplace=True)
#         df_list.append(df)
#     return pd.concat(df_list,axis=1).reset_index()

def quantiles(dataframe,grouping_by_columns,grouped_columns):
    grouped = dataframe.groupby(grouping_by_columns,dropna=False)[grouped_columns].agg([np.min, np.mean, np.max]).reset_index()
    grouped.columns = grouping_by_columns + [f"{prefix}_{agg_name}" for (prefix, agg_name) in grouped.columns if prefix not in grouping_by_columns]
    
    return grouped

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

def modify_investment(x,flood_protection_column,investment_column):
    if x[flood_protection_column] == 0:
        return 0
    else:
        return x[investment_column]

def change_string_to_array(x,column_name):
    values = x[column_name][1:-1].split(' ')
    v = []
    for val in values:
        if len(val) > 0:
            v.append(float(val))

    return (np.array(v))

def add_rows_and_transpose(df,df_merge,grouping_by_columns,grouped_columns,merge_df_columns):
    gr_cols = [c for c in df.columns.values.tolist() if c in grouped_columns]
    if len(grouping_by_columns) > 0:
        df = df.drop_duplicates(subset=grouping_by_columns,keep='first')
        df = grouping_values(df,grouping_by_columns,gr_cols)
    else:
        df = df[gr_cols].sum(axis=0)
        df = df.reset_index()
        grouping_by_columns = df.index
    
    df = df.melt(id_vars=grouping_by_columns, 
                var_name="key", 
                value_name="totals")
    df = pd.merge(df,df_merge,how="left",on=["key"]).fillna(0)
    df = df[grouping_by_columns + merge_df_columns + ["totals"]]
    
    return df

def group_and_write_results(dataframe,
                        stats_grouping_by,stats_writer,
                        quantile_grouping_by,quantile_writer,
                        grouped,sheetname,
                        add_values=True,write_values=True,generate_groups=True,generate_quantiles=True):
    if write_values == True:
        if generate_groups is True:
            ds = grouping_values(dataframe,stats_grouping_by,grouped,add_values=add_values) 
        else:
            ds = dataframe.copy()                   

        ds.to_excel(stats_writer,sheetname, index=False)
        # stats_writer.save()
        if generate_quantiles is True: 
            if add_values == False:
                grouped = ['totals'] 
            ds = quantiles(ds,quantile_grouping_by,grouped)
            ds.to_excel(quantile_writer,sheetname, index=False)
            # quantile_writer.save()

def get_risk_results(filedir,file,sector,sector_id,
                climate_columns,stats_comb,stats_wrtr,
                quantile_comb,quantile_wrtr,write_values=True):
    if (file.endswith("_risks.parquet")) and ('._' not in file):
        if "no_protection" in file:
            sh_name = 'no_protection'
            pr_name = 'no_protection_rp'
        elif 'design_protection' in file:
            sh_name = 'design'
            pr_name = 'design_protection_rp'
        else:
            fn = file.split('_to_')
            sh_name = f"{fn[0].split('_')[-1]}-{fn[1].split('_')[0]}"
            pr_name = f"{fn[0].split('_')[-1]}_to_{fn[1].split('_')[0]}_year_protection"
        
        exposures = pd.read_parquet(os.path.join(filedir,file))
        
        ead_cols = [c for c in exposures.columns.values.tolist() if "EAD_" in c]
        eael_cols = [c for c in exposures.columns.values.tolist() if "EAEL_" in c]
        index_cols = [c for c in exposures.columns.values.tolist() if c not in ead_cols + eael_cols]


        for idx,(el,el_cols) in enumerate([('EAD',ead_cols),('EAEL',eael_cols)]):
            risks = exposures[index_cols + el_cols].melt(id_vars=index_cols, 
                                var_name="key", 
                                value_name=f"{el}_{pr_name}")
            group = [f"{el}_{pr_name}"]

            el_keys = [(c.replace(f"{el}_","").replace(f"_{pr_name}",""),c) for c in el_cols]
            el_keys = [tuple(c[0].split("_") + [c[1]]) for c in el_keys]
            keys_df = pd.DataFrame(el_keys,columns=climate_columns+["key"])
            risks = pd.merge(risks,keys_df,how="left",on=["key"])
            risks = risks.drop_duplicates(subset=[sector_id] + stats_comb[f'{el}_groupby'],keep='first')
            group_and_write_results(risks,
                stats_comb[f'{el}_groupby'],stats_wrtr,
                quantile_comb['groupby'],quantile_wrtr,
                group,f"{sector}-{el}-{sh_name}",
                add_values=True,write_values=write_values,
                generate_groups = True,
                generate_quantiles=stats_comb['generate_quantiles'])

def get_cost_benefit_results(asset_risks,filedir,file,sector,sector_id_column,stats_comb,quantile_comb):
    if (file.endswith("_npvs.parquet")) and ('._' not in file):
        if "no_protection" in file:
            sh_name = 'no_protection'
            pr_name = 'no_protection_rp'
        elif 'design_protection' in file:
            sh_name = 'design'
            pr_name = 'design_protection_rp'
        else:
            fn = file.split('_to_')
            sh_name = f"{fn[0].split('_')[-1]}-{fn[1].split('_')[0]}"
            pr_name = f"{fn[0].split('_')[-1]}_to_{fn[1].split('_')[0]}_year_protection"
        
        exposures = pd.read_parquet(os.path.join(filedir,file))

        rcps = list(set(exposures.rcp.values.tolist()))
        ead_cols = [c for c in exposures.columns.values.tolist() if "EAD" in c]
        eael_cols = [c for c in exposures.columns.values.tolist() if "EAEL" in c]
        cost_cols = [c for c in exposures.columns.values.tolist() if "adapt_cost" in c or "cost_npv" in c]
        index_cols = [c for c in exposures.columns.values.tolist() if c not in ead_cols + eael_cols + cost_cols]
        
        for idx, (el,el_cols) in enumerate([("EAD",ead_cols),("EAEL",eael_cols)]):
            risks = exposures[
                        index_cols + el_cols
                        ].drop_duplicates(subset=[sector_id_column] + stats_comb[f'{el}_groupby'],keep='first')
            risks.rename(columns=dict([(c,f"{c}_{pr_name}_npv") for c in el_cols]),inplace=True)
            group = [f"{c}_{pr_name}_npv" for c in el_cols] 

            # print (risks)
            ar = quantiles(risks,[sector_id_column,pr_name] + quantile_comb['groupby'],group)

            if len(asset_risks) > 0:
                if pr_name in asset_risks.columns.values.tolist():
                    ar.drop(pr_name,axis=1,inplace=True)
                asset_risks = pd.merge(asset_risks,ar,how='left',on=[sector_id_column] + quantile_comb['groupby'])
            else:
                asset_risks = ar.copy()
        # print (asset_risks)
        for cost in cost_cols:
            if "ini_adapt_cost" in cost:
                costs = exposures[[sector_id_column, 
                                pr_name, 
                                cost]].drop_duplicates(subset=[sector_id_column, pr_name],
                                keep='first')
                costs.rename(columns={cost:f'{cost}_{pr_name}'},inplace=True)
            else:
                costs = exposures[[sector_id_column, 
                                    pr_name,
                                    cost] + stats_comb['cost_groupby']].drop_duplicates(
                                    subset=[sector_id_column, pr_name] + stats_comb['cost_groupby'],
                                    keep='first')
                costs.rename(columns={cost:f'{cost}_{pr_name}'},inplace=True)
                costs = quantiles(costs,[sector_id_column,pr_name],[f'{cost}_{pr_name}'])
            costs.drop(pr_name,axis=1,inplace=True)

            # costs = exposures[
            #                 index_cols + cost_cols
            #                 ].drop_duplicates(subset=[sector_id_column, pr_name] + quantile_comb['groupby'],keep='first')
            # costs.rename(columns={'median_total_maintenance_cost_npv':f'median_total_maintenance_cost_npv_{pr_name}',
            #                     'median_total_adapt_cost_npv':f'median_total_adapt_cost_npv_{pr_name}'},inplace=True)
            # group = [f'median_ini_adapt_cost_{pr_name}',
            #         f'median_total_maintenance_cost_npv_{pr_name}',
            #         f'median_total_adapt_cost_npv_{pr_name}']

            # print (costs.columns.values.tolist())
            asset_risks = pd.merge(asset_risks,
                            costs,
                            how='left',
                            on=[sector_id_column])
        # print (asset_risks)
    return asset_risks


def get_risk_timeseries_results(filedir,file,filestring,
                    sector,sector_id_column,start_year,end_year,
                    stats_comb,stats_wrtr,
                    quantile_comb,quantile_wrtr,write_values=True):
    for el in ['EAD','EAEL']:
        # print (f"{el.lower()}{filestring}.parquet")
        if (file.endswith(f"{el.lower()}_{filestring}.parquet")) and ('._' not in file):
            if "no_protection" in file:
                sh_name = 'no_protection'
                pr_name = 'no_protection_rp'
            elif 'design_protection' in file:
                sh_name = 'design'
                pr_name = 'design_protection_rp'
            else:
                fn = file.split('_to_')
                sh_name = f"{fn[0].split('_')[-1]}-{fn[1].split('_')[0]}"
                pr_name = f"{fn[0].split('_')[-1]}_to_{fn[1].split('_')[0]}_year_protection"
            
            exposures = pd.read_parquet(os.path.join(filedir,file))
            year_cols = [str(y) for y in np.arange(start_year,end_year+1,1)]

            if filestring == "_timeseries":
                group = f"{el}_{pr_name}_timeseries_scaled"
            else:
                group = f"{el}_{pr_name}_timeseries_npv"
            index_cols = [sector_id_column] + stats_comb[f'{el}_groupby']
            risks = exposures[index_cols + year_cols].drop_duplicates(subset=index_cols,keep='first')
            risks = risks.melt(id_vars= index_cols,var_name='epoch',value_name=group)
            group_and_write_results(risks,
                stats_comb[f'{el}_groupby'] + ['epoch'],stats_wrtr,
                quantile_comb['groupby'],quantile_wrtr,
                [group],f"{sector}-{el}-{sh_name}",
                add_values=True,write_values=write_values,
                generate_groups = True,
                generate_quantiles=stats_comb['generate_quantiles'])

def get_risk_timeseries_assets(filedir,file,filestring,
                    sector,sector_id_column,start_year,end_year,
                    stats_comb,stats_wrtr,
                    quantile_comb,quantile_wrtr,write_values=True):
    for el in ['EAD','EAEL']:
        # print (f"{el.lower()}{filestring}.parquet")
        if (file.endswith(f"{el.lower()}_{filestring}.parquet")) and ('._' not in file):
            if "no_protection" in file:
                sh_name = 'no_protection'
                pr_name = 'no_protection_rp'
            elif 'design_protection' in file:
                sh_name = 'design'
                pr_name = 'design_protection_rp'
            else:
                fn = file.split('_to_')
                sh_name = f"{fn[0].split('_')[-1]}-{fn[1].split('_')[0]}"
                pr_name = f"{fn[0].split('_')[-1]}_to_{fn[1].split('_')[0]}_year_protection"
            
            exposures = pd.read_parquet(os.path.join(filedir,file))
            year_cols = [str(y) for y in np.arange(start_year,end_year+1,1)]

            if filestring == "_timeseries":
                group = f"{el}_{pr_name}_timeseries_scaled"
            else:
                group = f"{el}_{pr_name}_timeseries_npv"
            index_cols = [sector_id_column] + stats_comb[f'{el}_groupby']
            # print (exposures)
            risks = exposures[index_cols + year_cols].drop_duplicates(subset=index_cols,keep='first')
            risks = risks.groupby([sector_id_column] + stats_comb['groupby'])[year_cols].mean().reset_index()
            # group_and_write_results(risks,
            #     stats_comb[f'{el}_groupby'] + ['epoch'],stats_wrtr,
            #     quantile_comb['groupby'],quantile_wrtr,
            #     [group],f"{sector}-{el}-{sh_name}",
            #     add_values=True,write_values=write_values,
            #     generate_groups = True,
            #     generate_quantiles=stats_comb['generate_quantiles'])

            group_and_write_results(risks,
                None,stats_wrtr,
                None,None,
                None,f"{sector}-{el}-{sh_name}",
                add_values=False,write_values=write_values,
                generate_groups=False,
                generate_quantiles=False)

# def get_risk_timeseries_assets(filedir,file,
#                     sector,start_year,end_year,
#                     stats_comb,stats_wrtr,
#                     write_values=True):
#     if (file.endswith("adaptation.csv")) and ('._' not in file):
#         if 'designed_protection' in file:
#             sh_name = 'design'
#             pr_name = 'designed_protection'
#         else:
#             fn = file.split('_to_')                                    
#             sh_name = f"{fn[0].split('_')[-1]}-{fn[1].split('_')[0]}"
#             pr_name = f"{fn[0].split('_')[-1]}_to_{fn[1].split('_')[0]}_year_protection"
#         if sector['results_folder_type'] == 'Zip':
#             exposures = pd.read_csv(filedir.open(file))
#         else:
#             exposures = pd.read_csv(os.path.join(filedir,file))
        
#         for el in ['EAD','EAEL']:
#             if 'year' in stats_comb[f'{el}_groupby']:
#                 stats_comb[f'{el}_groupby'].remove('year')
            
#             risks = exposures.drop_duplicates(subset=[sector['id_column']] + stats_comb[f'{el}_groupby'],keep='first')
#             if el == 'EAD':
#                 group = f"{el}_{pr_name}_timeseries_scaled"
#             else:
#                 group = f"{el}_{pr_name}_timeseries_growth"


#             ex = risks[[sector['id_column'],group] + stats_comb[f'{el}_groupby']]
#             ex[group] = ex.progress_apply(lambda x: change_string_to_array(x,group), axis=1)
#             ex[[str(y) for y in np.arange(start_year,end_year+1,1)]] = ex[group].apply(pd.Series)
#             ex.drop([group],axis=1,inplace=True)
#             cols = [str(y) for y in np.arange(start_year,end_year+1,1)]
#             ex = ex.groupby([sector['id_column']] + stats_comb['groupby'])[cols].quantile(0.5).reset_index()
            

#             group_and_write_results(ex,
#                 None,stats_wrtr,
#                 None,None,
#                 None,f"{sector['sector']}-{el}-{sh_name}",
#                 add_values=False,write_values=write_values,
#                 generate_groups=False,
#                 generate_quantiles=False)