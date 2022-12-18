# -*- coding: utf-8 -*-
"""Preprocess utils
"""
import os
import sys
import json
import pandas as pd
import geopandas as gpd
from geopy import distance
import shapely.geometry
from shapely.geometry import Point,LineString
from boltons.iterutils import pairwise
import geopandas
from tqdm import tqdm
tqdm.pandas()

def load_config():
    """Read config.json
    """
    config_path = os.path.join(os.path.dirname(__file__), '..','..','config.json')
    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    return config

def line_length_km(line, ellipsoid='WGS-84'):
    """Length of a line in meters, given in geographic coordinates.

    Adapted from https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

    Args:
        line: a shapely LineString object with WGS-84 coordinates.

        ellipsoid: string name of an ellipsoid that `geopy` understands (see http://geopy.readthedocs.io/en/latest/#module-geopy.distance).

    Returns:
        Length of line in kilometers.
    """
    if line.geometryType() == 'MultiLineString':
        return sum(line_length_km(segment) for segment in line)

    return sum(
        distance.distance(tuple(reversed(a)), tuple(reversed(b)),ellipsoid=ellipsoid).km
        for a, b in pairwise(line.coords)
    )

def nearest_geom(x,gdf,id_column):
    nearest_index = gdf.distance(x).sort_values().index[0]
    return gdf.loc[nearest_index][id_column]

def match_assets_to_boundaries(asset_shape,boundary_shape,asset_id_column,boundary_column,asset_type="nodes"):
    matches = gpd.sjoin(asset_shape,
                        boundary_shape[[boundary_column,'geometry']], 
                        how="inner", predicate='intersects').reset_index()

    unique_matches = list(set(matches[asset_id_column].values.tolist()))
    no_matches = []
    if len(unique_matches) < len(asset_shape.index):
        no_matches = asset_shape[~asset_shape[asset_id_column].isin(unique_matches)]
        sindex_boundary = boundary_shape.sindex
        no_matches[boundary_column] = no_matches.geometry.progress_apply(
                                    lambda x: nearest_geom(x,
                                        boundary_shape,boundary_column)) 

    if asset_type != "node":
        if len(matches.index) > len(asset_shape.index):
            matches.rename(columns={"geometry":"asset_geometry"},inplace=True)
            matches = pd.merge(matches, boundary_shape[[boundary_column,'geometry']],how="left",on=[boundary_column])
            if asset_type == "edges":
                matches["geom_match"] = matches.progress_apply(lambda x:x["asset_geometry"].intersection(x["geometry"].buffer(0)).length,axis=1)
            else:
                matches["geom_match"] = matches.progress_apply(lambda x:x["asset_geometry"].intersection(x["geometry"].buffer(0)).area,axis=1)
            matches = matches.sort_values(by=["geom_match"],ascending=False)
            matches = matches.drop_duplicates(subset=[asset_id_column], keep="first")
            matches.drop(["geometry","geom_match"],axis=1,inplace=True)
            matches.rename(columns={"asset_geometry":"geometry"},inplace=True)

    if len(no_matches) > 0:
        matches = pd.concat([matches,no_matches],axis=0,ignore_index=True)

    return matches[asset_shape.columns.values.tolist() + [boundary_column]]

