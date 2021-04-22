import numpy as np
import pandas as pd
import geopandas as gpd
import os
import statistics

import seaborn as sn
import streamlit as st
import pydeck as pdk
import shapely.wkt

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide", page_title="Ex-stream-ly Cool App",
                  page_icon="🧊", initial_sidebar_state="expanded",)

# @st.cache(persist=True, allow_output_mutation=True)
def load_gdf():
    def parse_geometry(x):
        return shapely.wkt.loads(x["geometry"])
    def parse_columns(x):
        from ast import literal_eval
        x['path'] = literal_eval(x['path'])
        x['color_lottr'] = tuple(map(float, x['color_lottr'].strip('()').split(', ')))
        return x
    df = pd.read_csv(os.path.join(os.getcwd(), 'advise_2021Q1.csv'))
    df = df.apply(parse_columns, axis=1)
    geometry = df.apply(parse_geometry, axis=1)
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)
    return gdf


st.title("Exploring Advisè Corridor Reliability")

row1_1, row1_2 = st.beta_columns((2,10))

with row1_1:
    with st.beta_container():
        region_radio = st.radio(
            "Select region to Zoom in:",
            ("Tennessee", "Knoxville", "Chattanooga", "Nashville", "Memphis"),
            index=0
        )
        time_radio = st.radio(
            "Select time period to display:",
            ("January", "Feburary", "March", "Quarter 1"),
            index=0
        )
        type_radio = st.radio(
            "Select type of data to report:",
            ("LOTTR", "TTTR", "Excessive Dalay"),
            index=0
        )

def get_prefix_by_time(time_radio):
    if time_radio == "January":
        return 'Jan.'
    elif time_radio == "Februray":
        return 'Feb.'
    elif time_radio == "March":
        return 'Mar.'
    else:
        return 'Q1'

def get_suffix_by_type(type_radio):
    if type_radio == "LOTTR":
        return "_lottr"
    if type_radio == "TTTR":
        return "_tttr"
    if type_radio == "Excessive Dalay":
        return "_ed"

def filt_df(df, region_radio):
    if region_radio == 'Tennessee':
        return df.copy()
    elif region_radio == 'Knoxville':
        code = 1
    elif region_radio == 'Chattanooga':
        code = 2
    elif region_radio == 'Nashville':
        code = 3
    else: # Memphis
        code = 4
    filt = (df.met_region == code)
    df = df[filt].copy()
    return df

def find_best_view(df, region_radio):
    mid_lats, mid_longs = df.MidLat, df.MidLong
    m_lat = statistics.median(mid_lats)
    m_long = statistics.median(mid_longs)
    zoom = 7.5
    if region_radio == 'Tennessee':
        zoom = 6
    return (m_lat, m_long, zoom)


def get_map(df, lat, lon, zoom, region_radio, time_radio, type_radio):
    prefix_n = get_prefix_by_time(time_radio)
    suffix_n = get_suffix_by_type(type_radio)
    col_name = prefix_n + suffix_n
    def get_color(x, col_name, xmin, xmax):
        import matplotlib
        import matplotlib.cm as cm
        norm = matplotlib.colors.Normalize(vmin=xmin, vmax=xmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
        if np.isnan(x[col_name]):
            return (100.0,100.0,100.0)
        txt = list(mapper.to_rgba(x[col_name])[:-1])
        txt = tuple(255*item for item in txt)
        return txt
    # update color by the specified lottr
    df['color_lottr'] = df.apply(get_color, axis=1, args=(col_name, 1, 1.5))
    
    path_layer = pdk.Layer(
        type="PathLayer",
        data=df,
        get_path="path",
        # get_color=GET_COLOR_JS,
        # width_scale=10,
        width_min_pixels=4,
        get_color="color_lottr",
        auto_highlight=True,
        pickable=True,
        filled=True,
        opacity=0.8
    )
    # Set the viewport location
    view_state = pdk.ViewState(
        longitude=lon,
        latitude=lat,
        zoom=zoom,
        min_zoom=5,
        max_zoom=15,
        pitch=0,
        bearing=0)
    # set tooltip
    tooltip = {'html': '<b>Elevation Value:</b> '+type_radio+': {elevation}',
               'style': {'color': 'white'}
              }
    return pdk.Deck(layers=[path_layer], initial_view_state=view_state,
                     map_style=pdk.map_styles.ROAD, tooltip = tooltip)

gdf = load_gdf()
gdf = filt_df(gdf, region_radio)
view_p = find_best_view(gdf, region_radio)
my_map = get_map(gdf, view_p[0], view_p[1], view_p[2],
                 region_radio, time_radio, type_radio)

with row1_2:
    st.write(my_map)








