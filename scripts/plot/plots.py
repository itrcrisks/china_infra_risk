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
    plot_region_names = True # This will plot region names in the Airport maps. Set it to False if you do not want that
    length_division = 1000.0 # Convert m to km

    duration = 10
    ead_column_d = 'max_EAD_designed_protection'
    eael_column_d = 'max_EAEL_designed_protection'
    ead_column_ud = 'max_EAD_undefended'
    eael_column_ud = 'max_EAEL_undefended'

    """Specify the data paths for the project and its subfolders 
        Create new folders for the results within this path 
    """
    data_path = "/Users/raghavpant/Desktop/china_study"
    network_path = os.path.join(data_path,'network') # Where we have all the network shapefiles
    figures = os.path.join(data_path,'figures')
    if os.path.exists(figures) == False:
        os.mkdir(figures)
    stats = os.path.join(data_path,'stats')
    if os.path.exists(stats) == False:
        os.mkdir(stats)

    """Step 1: Process results of each type 
    """

    df = pd.read_csv(os.path.join(data_path,'asset_details','air_intersection_cost_final.csv'))
    print (df)
    s = df.groupby([
                    'climate_scenario',
                    'model',
                    'year',
                    'probability']
                    )['ID'].agg(lambda x:len(x.unique())).reset_index(name='Number of Assets')
    print (s)

    """Step 2: Plot
    """
    group_column = 'probability'
    length_column = 'Number of Assets'
    current_year = 2016
    current_scenario = 'none'
    current_climate = s[(s.year == current_year) & (s.climate_scenario == current_scenario)]
    future_climate = s[s['year'] > current_year]
    climate_scenarios_years = list(set(list(zip(future_climate['climate_scenario'].values.tolist(),
    											future_climate['year'].values.tolist()))))

    for cl_y in climate_scenarios_years:
        flooding_scenario = future_climate[(future_climate.year == cl_y[1]) & (future_climate.climate_scenario == cl_y[0])] 
        if len(flooding_scenario.index) > 0:
            flooding_scenario_mean = flooding_scenario.groupby([group_column])[length_column].quantile(0.5).reset_index()
            flooding_scenario_min = flooding_scenario.groupby([group_column])[length_column].quantile(0.05).reset_index()
            flooding_scenario_max = flooding_scenario.groupby([group_column])[length_column].quantile(0.95).reset_index()                  

if __name__ == '__main__':
    main()
