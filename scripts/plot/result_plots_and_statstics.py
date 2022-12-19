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
from map_plotting import *
import seaborn as sns

def get_nearest_node(x, sindex_input_nodes, input_nodes, id_column):
    """Get nearest node in a dataframe

    Parameters
    ----------
    x
        row of dataframe
    sindex_nodes
        spatial index of dataframe of nodes in the network
    nodes
        dataframe of nodes in the network
    id_column
        name of column of id of closest node

    Returns
    -------
    Nearest node to geometry of row
    """
    return input_nodes.loc[list(sindex_input_nodes.nearest(x.bounds[:2]))][id_column].values[0]

def extract_gdf_values_containing_nodes(x, sindex_input_gdf, input_gdf, column_name):
    a = input_gdf.loc[list(input_gdf.geometry.contains(x.geometry))]
    if len(a.index) > 0:
        return a[column_name].values[0]
    else:
        return get_nearest_node(x.geometry, sindex_input_gdf, input_gdf, column_name)

def spatial_intersection_points(point_dataframe,polygon_dataframe,point_id_column,polygon_column):
    sindex_poly = polygon_dataframe.sindex
    point_matches = gpd.sjoin(point_dataframe,
                        polygon_dataframe[[polygon_column,'geometry']], how="inner", op='within').reset_index()
    no_points = [x for x in point_dataframe[point_id_column].tolist() if x not in point_matches[point_id_column].tolist()]

    if no_points:
        remain_points = point_dataframe[point_dataframe[point_id_column].isin(no_points)]
        remain_points[polygon_column] = remain_points.apply(lambda x: extract_gdf_values_containing_nodes(
            x, sindex_poly, polygon_dataframe,polygon_column), axis=1)

        point_matches = pd.concat([point_matches,remain_points],axis=0,sort='False', ignore_index=True)

    return point_matches[[point_id_column,polygon_column]]

def spatial_intersections_lines(line_dataframe, 
                    polygon_dataframe, 
                    line_id_column,polygon_column):
    """Intersect network edges/nodes and boundary Polygons to collect boundary and hazard attributes

    Parameters
        - network_shapefile - Shapefile of edge LineStrings or node Points
        - polygon_shapefile - Shapefile of boundary Polygons
        - network_type - String value -'edges' or 'nodes' - Default = 'nodes'
        - name_province - String name of province if needed - Default = ''

    Outputs
        data_dictionary - Dictionary of network-hazard-boundary intersection attributes:
            - edge_id/node_id - String name of intersecting edge ID or node ID
            - length - Float length of intersection of edge LineString and hazard Polygon: Only for edges
            - province_id - String/Integer ID of Province
            - province_name - String name of Province in English
            - district_id - String/Integer ID of District
            - district_name - String name of District in English
            - commune_id - String/Integer ID of Commune
            - commune_name - String name of Commune in English
            - hazard_attributes - Dictionary of all attributes from hazard dictionary
    """
    line_gpd = line_dataframe
    poly_gpd = polygon_dataframe
    data_dictionary = []

    if len(line_gpd.index) > 0 and len(poly_gpd.index) > 0:
        # create spatial index
        poly_sindex = poly_gpd.sindex
        for l_index, lines in line_gpd.iterrows():
            intersected_polys = poly_gpd.iloc[list(
                poly_sindex.intersection(lines.geometry.bounds))]
            for p_index, poly in intersected_polys.iterrows():
                if (lines['geometry'].intersects(poly['geometry']) is True) and (poly.geometry.is_valid is True) and (lines.geometry.is_valid is True):
                    data_dictionary.append({line_id_column: lines[line_id_column],
                                            polygon_column: poly[polygon_column]})

    return pd.DataFrame(data_dictionary)

def main():
    """Specify all the inputs we will be needing
    """
    sector_descriptions = [
                        {
                            'sector':'road',
                            'sector_label':'Roads',
                            'id_column':'road_ID',
                            'asset_type':'edges',
                            'sector_shapefile':'final_road.shp',
                            'sector_marker':None,
                            'sector_size':1.0,
                            'sector_color':'#969696',
                        },
                        {
                            'sector':'landfill',
                            'sector_label':'Landfill sites',
                            'id_column':'Plant_Numb',
                            'asset_type':'nodes',
                            'sector_shapefile':'landfill_nodes.shp',
                            'sector_marker':'o',
                            'sector_size':12.0,
                            'sector_color':'#74c476',
                        },
                        {
                            'sector':'air',
                            'sector_label':'Airports',
                            'id_column':'ID',
                            'asset_type':'nodes',
                            'sector_shapefile':'air_nodes.shp',
                            'sector_marker':'s',
                            'sector_size':12.0,
                            'sector_color':'#636363',
                        },
                        {
                            'sector':'power',
                            'sector_label':'Power plants',
                            'id_column':'Number',
                            'asset_type':'nodes',
                            'sector_shapefile':'power_nodes.shp',
                            'sector_marker':'^',
                            'sector_size':12.0,
                            'sector_color':'#fb6a4a',
                        },
                        ]
    # We assume all our results are stored in folders 
    # within which we speficy the files names as per a convention
    # string_1_{}_string_2 where the {} is the sector key of sector_descriptions
    # This is how we had created the results so it should be the case                   
    results_description = [
                            {
                            'type':'exposure',
                            'folder':'asset_details',
                            'string_1':'',
                            'string_2':'_intersection_cost_final.csv',
                            'figure_name':'china_asset_exposures.png'
                            },
                            {
                            'type':'risk',
                            'folder':'risk_results',
                            'string_1':'',
                            'string_2':'_risks.csv',
                            'figure_name':'china_asset_risks_defended.png'
                            },
                            {
                            'type':'adaptation',
                            'folder':'adaptation_results',
                            'string_1':'output_adaptation_',
                            'string_2':'_10_days_max_asset_dictionary_with_projected_growth_4p5_discount_rates.csv',
                            'figure_name':'china_asset_investments_avoidedrisks_defended.png'
                            }
                            ]
    figure_texts = ['a.','b.','c.','d.','e.','f.','g.','h.'] # Labels for 8 subplots in a figure
    flood_divisions = 4 # We have decided to create 4 scales for values in our results 
    flood_colors_graded = ['#c6dbef','#6baed6','#2171b5','#08306b'] # For different intensities of values in maps
    flood_colors_graded = ['#4292c6','#2171b5','#08519c','#08306b'] # For different intensities of values in maps
    flood_colors_graded = ['#7fcdbb','#41b6c4','#2c7fb8','#253494'] # For different intensities of values in maps
    change_colors = ['#d73027','#fc8d59','#fee08b','#d9ef8b','#91cf60','#1a9850','#969696']
    change_labels = ['0%','>0% - 20%','20% - 40%','40% - 60%','60% - 80%','80% - 100%','No value']
    change_ranges = [(-1,0),(0,20),(20,40),(40,60),(60,80),(80,100),(-2,-1)]
    """Note: If you want more than 4 scales on your maps then
        Set flood_divisions to the new value (integer only)
        If you set flood_divisions > 4 then you have to expand the flood_colors_graded to include more colors
        For more colors see: https://colorbrewer2.org/
    """
    flood_color = '#3182bd' # When all flood results are shown with only one color
    noflood_color = '#969696' # For no flooding result cases on maps
    plot_region_names = True # This will plot region names inn the Airport maps. Set it to False if you do not want that
    length_division = 1000.0 # Convert m to km

    duration = 10
    ead_column_d = 'max_EAD_designed_protection'
    eael_column_d = 'max_EAEL_designed_protection'
    ead_column_ud = 'max_EAD_undefended'
    eael_column_ud = 'max_EAEL_undefended'

    """Specify the data paths for the project and its subfolders 
        Create new folders for the results within this path 
    """
    data_path = "/Users/raghavpant/Desktop/sisi_python"
    network_path = os.path.join(data_path,'network') # Where we have all the network shapefiles
    figures = os.path.join(data_path,'figures')
    if os.path.exists(figures) == False:
        os.mkdir(figures)
    stats = os.path.join(data_path,'stats')
    if os.path.exists(stats) == False:
        os.mkdir(stats)


    """
    Step 1: Get all the Administrative boundary files
        Combine the English provinces names with the Chinese ones  
    """
    boundary_gpd = gpd.read_file(os.path.join(data_path,
                                'province_shapefile',
                                'China_pro_pop_electricity.shp'),encoding="utf-8")
    admin_gpd = gpd.read_file(os.path.join(data_path,
                                'Household_shapefile',
                                'County_pop_2018.shp'),encoding="utf-8")
    boundary_names = pd.merge(admin_gpd[['AdminCode','ProAdCode']],
                    boundary_gpd[['DZM','Region']],
                    how='left',
                    left_on=['ProAdCode'],
                    right_on=['DZM'])
    boundary_names['AdminCode']=boundary_names['AdminCode'].astype(int)
    
    """
    Step 2: Get map bounds to set the map projections  
    """
    bounds = boundary_gpd.geometry.total_bounds # this gives your boundaries of the map as (xmin,ymin,xmax,ymax)
    ax_proj = get_projection(extent = (bounds[0]+5,bounds[2]-10,bounds[1],bounds[3]))
    del admin_gpd


    """Step 3: Process results of each type 
    """

    for result in results_description:
        fig, ax_plots = plt.subplots(2,4,
                            subplot_kw={'projection': ax_proj},
                            figsize=(24,10),
                            dpi=500)
        ax_plots = ax_plots.flatten()
        j = 0
        for s in range(len(sector_descriptions)):
            legend_handles = []
            z_order = 5
            sector = sector_descriptions[s]
            if sector['sector'] == 'air' and plot_region_names == True:
                ax = plot_basemap(ax_plots[j],include_labels=True)
            else:
                ax = plot_basemap(ax_plots[j],include_labels=False)

            assets = gpd.read_file(os.path.join(network_path,
                                sector['sector_shapefile']))
            
            if 'Region' in assets.columns.values.tolist():
                    assets.drop(['Region'],
                                    axis=1,
                                    inplace=True)
            results_csv = pd.read_csv(os.path.join(data_path,
                                    result['folder'],'{}{}{}'.format(result['string_1'],
                                                                sector['sector'],
                                                                result['string_2'])))

            if result['type'] == 'exposure':
                if 'Region' in results_csv.columns.values.tolist():
                    results_csv.drop(['Region'],
                                    axis=1,
                                    inplace=True)
                if 'probability' in results_csv.columns.values.tolist():
                    results_csv['return_period'] = 1/results_csv['probability']
                if 'AdminCode' in results_csv.columns.values.tolist():
                    results_csv['AdminCode'] = results_csv['AdminCode'].astype(int)
                    results_csv = pd.merge(results_csv,
                                            boundary_names,
                                            how='left',
                                            on=['AdminCode'])
                elif 'AdminCode_x' in results_csv.columns.values.tolist():
                    results_csv['AdminCode_x'] = results_csv['AdminCode_x'].astype(int)
                    results_csv = pd.merge(results_csv,
                                        boundary_names,
                                        how='left',
                                        left_on=['AdminCode_x'],
                                        right_on=['AdminCode'])
                s = results_csv.groupby(['Region',
                                    'climate_scenario',
                                    'model',
                                    'year']
                                    )[sector['id_column']].agg(lambda x:len(x.unique())).reset_index(name='Number of Assets')
                s.to_csv(os.path.join(stats,
                        '{}_exposure_numbers_by_region_climate_scenario_model_year.csv'.format(sector['sector'])))

                s = results_csv.groupby(['Region',
                                    'climate_scenario',
                                    'model',
                                    'year',
                                    'return_period']
                                    )[sector['id_column']].agg(lambda x:len(x.unique())).reset_index(name='Number of Assets')
                s.to_csv(os.path.join(stats,
                        '{}_exposure_numbers_by_region_climate_scenario_model_year_return_period.csv'.format(sector['sector'])))

                if sector['asset_type'] == 'edges':
                    results_csv['length'] = 1.0*results_csv['length']/length_division
                    s = results_csv.groupby(['Region',
                                    'climate_scenario',
                                    'model',
                                    'year',
                                    'return_period']
                                    )['length'].sum().reset_index()
                    s.to_csv(os.path.join(stats,
                        '{}_exposure_length_by_region_climate_scenario_model_year_return_period.csv'.format(sector['sector'])))

                flood_ids = list(set(results_csv[results_csv['year'] == 2016][sector['id_column']].values.tolist()))

                for i in range(0,2):
                    if i == 0:
                        plot_assets = assets[~assets[sector['id_column']].isin(flood_ids)]
                        color = noflood_color
                    else:
                        plot_assets = assets[assets[sector['id_column']].isin(flood_ids)]
                        color = flood_color

                    if sector['asset_type'] == 'nodes':
                        ax = plot_point_assets(ax,plot_assets,color,
                                            sector['sector_size'],
                                            sector['sector_marker'],
                                            z_order)
                    else:
                        ax = plot_line_assets(ax,plot_assets,color,
                                            sector['sector_size'],
                                            z_order)
                    z_order += 1
                legend_handles.append(mpatches.Patch(color=flood_color,
                                        label='Flood exposure'))
                legend_handles.append(mpatches.Patch(color=noflood_color,
                                        label='No flood exposure'))

                ax.legend(handles=legend_handles,fontsize=10,loc='lower left')
                ax.text(
                    0.05,
                    0.95,
                    figure_texts[j],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=14,
                    weight='bold')
                ax.text(
                    0.35,
                    0.95,
                    sector['sector_label'],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=14,
                    weight='bold')
                
                ax_plots[j+4].remove()
                ax = fig.add_subplot(2, 4, j+5)
                s = results_csv.groupby(['model', 
                                    'climate_scenario',
                                    'year'])[sector['id_column']].agg(lambda x:len(x.unique())).reset_index(name='Number of Assets')
                s['percentage'] = 100.0*s['Number of Assets']/len(assets[sector['id_column']])
                sns.boxplot(y='percentage', x='year', 
                     data=s, 
                     width=0.2,
                     palette="Blues",showfliers = False,ax= ax)
                ax.set_xlabel('Year')
                ax.set_ylabel('Assets exposed (%)')
                ax.text(
                    0.05,
                    0.95,
                    figure_texts[j+4],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=14,
                    weight='bold')

                j+=1
    
            elif result['type'] == 'risk':
                results_csv['risk'] = results_csv[ead_column_d] + duration*results_csv[eael_column_d]
                results_csv['risk'] = results_csv['risk']/1e6

                if sector['asset_type'] == 'nodes':
                    region_matches = spatial_intersection_points(assets,
                                                        boundary_gpd,
                                                        sector['id_column'],
                                                        'Region')
                else:
                    region_matches = spatial_intersections_lines(assets,
                                                        boundary_gpd,
                                                        sector['id_column'],
                                                        'Region')
                results_csv = pd.merge(results_csv,region_matches,how='left',on=[sector['id_column']])
                s = results_csv.groupby(['Region',
                            'climate_scenario',
                            'model',
                            'year']
                        )['risk'].max().reset_index(name='Max. Risk RMB Millions')
                s.to_csv(os.path.join(stats,
                        '{}_max_risks_by_region_climate_scenario_model_year.csv'.format(sector['sector'])))

                assets = gpd.GeoDataFrame(pd.merge(assets[[sector['id_column'],'geometry']],
                                        results_csv[results_csv['year'] == 2016][[sector['id_column'],'risk']],
                                        how='left',on=[sector['id_column']]),
                                        geometry='geometry',crs='epsg:4326')
                assets['risk'].fillna(0,inplace=True)

                ax = plot_basemap(ax_plots[j],include_labels=False)
                # print (assets)
                if sector['asset_type'] == 'nodes':
                    ax = point_map_plotting_color_width(ax,assets,'risk',
                                        sector['sector_marker'],1.0,
                                        'Total risks (RMB million)','Risk',
                                        point_colors=flood_colors_graded
                                        )
                else:
                    ax = line_map_plotting_colors_width(ax,assets,'risk',
                        1.0,
                        'Total risks (RMB million)',
                        'Risk',
                        line_colors=flood_colors_graded,
                        width_step=0.04
                        )
                ax.text(
                    0.05,
                    0.95,
                    figure_texts[j],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=14,
                    weight='bold')
                ax.text(
                    0.35,
                    0.95,
                    sector['sector_label'],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=14,
                    weight='bold')
                
                ax_plots[j+4].remove()
                ax = fig.add_subplot(2, 4, j+5)
                s = results_csv.groupby([sector['id_column'],
                                    'year'])['risk'].max().reset_index()

                # s['risk'] = s['risk']/1e6
                sns.boxplot(y='risk', x='year', 
                     data=s, 
                     width=0.2,
                     palette="Blues",
                     showfliers = True,
                     flierprops = dict(marker='o', markersize=4,
                          linestyle='none'),
                     ax= ax)
                ax.set_xlabel('Year')
                ax.set_ylabel('Max. Assets Risks (RMB millions)')
                ax.text(
                    0.05,
                    0.95,
                    figure_texts[j+4],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=14,
                    weight='bold')

                j+=1
            elif result['type'] == 'adaptation':
                adapt_invest = results_csv.copy()
                adapt_invest['ini_investment'] = adapt_invest['ini_investment']/1e6
                adapt_invest = adapt_invest[adapt_invest['year'] == 2016]
                if sector['asset_type'] == 'nodes':
                    region_matches = spatial_intersection_points(assets,
                                                        boundary_gpd,
                                                        sector['id_column'],
                                                        'Region')
                else:
                    region_matches = spatial_intersections_lines(assets,
                                                        boundary_gpd,
                                                        sector['id_column'],
                                                        'Region')
                adapt_invest = pd.merge(adapt_invest,region_matches,how='left',on=[sector['id_column']])
                s_tot = adapt_invest.groupby(['Region']
                        )['ini_investment'].sum().reset_index(name='Total initial investment RMB Millions')
                s_max = adapt_invest.groupby(['Region']
                        )['ini_investment'].max().reset_index(name='Maximum initial investment RMB Millions')
                s_mean = adapt_invest.groupby(['Region']
                        )['ini_investment'].median().reset_index(name='Median initial investment RMB Millions')
                s = pd.merge(s_tot,s_max,how='left',on=['Region'])
                s = pd.merge(s,s_mean,how='left',on=['Region'])
                s.to_csv(os.path.join(stats,
                        '{}_investments_by_region.csv'.format(sector['sector'])))
                # adapt_invest = adapt_invest[adapt_invest['year'] == 2016]
                adapt_invest = adapt_invest[[sector['id_column'],'ini_investment']]
                # adapt_invest.drop_duplicates(subset=[sector['id_column']],keep='first',inplace=True)
                assets = gpd.GeoDataFrame(pd.merge(assets[[sector['id_column'],'geometry']],
                                        adapt_invest[[sector['id_column'],'ini_investment']],
                                        how='left',on=[sector['id_column']]),
                                        geometry='geometry',crs='epsg:4326')
                assets['ini_investment'].fillna(0,inplace=True)
                # assets['ini_investment'] = assets['ini_investment']/1e6

                ax = plot_basemap(ax_plots[j],include_labels=False)
                if sector['asset_type'] == 'nodes':
                    ax = point_map_plotting_color_width(ax,assets,'ini_investment',
                                            sector['sector_marker'],1.0,
                                            'Investment (RMB million)','Risk',
                                            point_colors=flood_colors_graded
                                            )
                else:
                    ax = line_map_plotting_colors_width(ax,assets,'ini_investment',
                        1.0,
                        'Investment (RMB million)',
                        'Risk',
                        line_colors=flood_colors_graded,
                        width_step=0.04
                        )
                ax.text(
                    0.05,
                    0.95,
                    figure_texts[j],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=14,
                    weight='bold')
                ax.text(
                    0.10,
                    0.95,
                    '{} - Adaptation investments'.format(sector['sector_label']),
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=11,
                    weight='bold')

                assets = gpd.read_file(os.path.join(data_path,'network',sector['sector_shapefile']))
                flood_risks = pd.read_csv(os.path.join(data_path,'risk_results','{}_risks.csv'.format(sector['sector'])))
                flood_risks['risk_d'] = flood_risks[ead_column_d] + duration*flood_risks[eael_column_d]
                flood_risks['risk_ud'] = flood_risks[ead_column_ud] + duration*flood_risks[eael_column_ud]
                flood_risks['a_risk'] = 100.0*(flood_risks['risk_ud'] - flood_risks['risk_d'])/flood_risks['risk_ud']
                assets = gpd.GeoDataFrame(pd.merge(assets[[sector['id_column'],'geometry']],
                                        flood_risks[flood_risks['year'] == 2016][[sector['id_column'],'a_risk']],
                                        how='left',on=[sector['id_column']]),
                                        geometry='geometry',crs='epsg:4326')
                assets['a_risk'].fillna(-1,inplace=True)
                ax = plot_basemap(ax_plots[j+1],include_labels=False)
                ch_results = []
                for ch in range(len(change_ranges)):
                    ch_assets = assets[(assets['a_risk'] > change_ranges[ch][0]) & (assets['a_risk'] <= change_ranges[ch][1])]
                    adapt_ch = pd.merge(ch_assets,region_matches,how='left',on=[sector['id_column']])
                    s = adapt_ch.groupby(['Region',
                                    ]
                                    )[sector['id_column']].agg(lambda x:len(x.unique())).reset_index(name='Number of Assets')
                    s['change_label'] = change_labels[ch]
                    # ch_results.append(s)
                    if len(ch_results) > 0:
                    	print (ch_results)
                    	print ('s',s)
                    	ch_results = pd.concat([ch_results,s],axis=0,sort='False', ignore_index=True)
                    else:
                    	ch_results = s.copy()

                    if sector['asset_type'] == 'nodes':
                        ax = plot_point_assets(ax,ch_assets,
                                            change_colors[ch],
                                            sector['sector_size'],
                                            sector['sector_marker'],
                                            z_order)
                        legend_handles.append(plt.plot([],[],
                                            marker=sector['sector_marker'], 
                                            ms=sector['sector_size'], 
                                            ls="",
                                            color=change_colors[ch],
                                            label=change_labels[ch])[0])
                    else:
                        ax = plot_line_assets(ax,ch_assets,
                                            change_colors[ch],
                                            sector['sector_size'],
                                            z_order)
                        legend_handles.append(mpatches.Patch(color=change_colors[ch],
                                                label=change_labels[ch]))


                ax.legend(handles=legend_handles,fontsize=8,loc='lower left')
                ax.text(
                    0.05,
                    0.95,
                    figure_texts[j+1],
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=14,
                    weight='bold')
                ax.text(
                    0.10,
                    0.95,
                    '{} - Avoided risks'.format(sector['sector_label']),
                    horizontalalignment='left',
                    transform=ax.transAxes,
                    size=11,
                    weight='bold')

                j+=2
                ch_results.to_csv(os.path.join(stats,'{}_region_avoided_risks_percentage_asset_numbers.csv'.format(sector['sector'])))
        plt.tight_layout()
        save_fig(os.path.join(figures,result['figure_name']))
        plt.close()


if __name__ == '__main__':
    main()
