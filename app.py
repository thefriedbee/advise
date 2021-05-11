import numpy as np
import pandas as pd
import geopandas as gpd
import os
import statistics

import seaborn as sn
import streamlit as st
import pydeck as pdk
import shapely.wkt
import altair as alt

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide", page_title="ADVISE Interactive Dashboard",
                  page_icon="🧊", initial_sidebar_state="expanded",)

@st.cache(persist=True, allow_output_mutation=True)
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


st.title("Exploring ADVISE Corridor Reliability")

row1_2, row1_3 = st.beta_columns((9,1))
row2_2, row2_3 = st.beta_columns((9,1))


with st.sidebar.beta_container():
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
        return 'Jan'
    elif time_radio == "Feburary":
        return 'Feb'
    elif time_radio == "March":
        return 'Mar'
    else:
        return 'Q1'

def get_suffix_by_type(type_radio):
    if type_radio == "LOTTR":
        return "_lottr"
    if type_radio == "TTTR":
        return "_tttr"
    if type_radio == "Excessive Dalay":
        return "_ed"

@st.cache(persist=True, allow_output_mutation=True)
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


@st.cache(persist=True, allow_output_mutation=True)
def find_best_view(df, region_radio):
    mid_lats, mid_longs = df.MidLat, df.MidLong
    m_lat = statistics.median(mid_lats)
    m_long = statistics.median(mid_longs)
    zoom = 7.5
    if region_radio == 'Tennessee':
        zoom = 6
    return (m_lat, m_long, zoom)

def plot_color_gradients(cmap, vmin, vmax):
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.8, 2))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
    cb = plt.colorbar(mappable=mapper, ax=ax, orientation='vertical')
    ax.remove()
    return fig

@st.cache(persist=True, allow_output_mutation=True)
def get_map(df, lat, lon, zoom, region_radio, time_radio, type_radio):
    prefix_n = get_prefix_by_time(time_radio)
    suffix_n = get_suffix_by_type(type_radio)
    col_name = prefix_n + suffix_n
    # prepare data
    df['elevation'] = df[col_name].replace(np.nan, 0)
    df['elevation'] = df['elevation'].apply(lambda x:round(x,2))
    def get_color(x, col_name, xmin, xmax):
        import matplotlib
        # import matplotlib.cm as cm
        norm = matplotlib.colors.Normalize(vmin=xmin, vmax=xmax, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.plasma)
        if np.isnan(x[col_name]):
            return (100.0,100.0,100.0)
        txt = list(mapper.to_rgba(x[col_name])[:-1])
        txt = tuple(255*item for item in txt)
        return txt
    # update color by the specified lottr
    df['color_lottr'] = df.apply(get_color, axis=1, args=(col_name, 1, 1.5))
    vmin, vmax = 1, 1.5
    if type_radio == "Excessive Dalay":
        df['color_lottr'] = df.apply(get_color, axis=1, args=(col_name, 0, 1000))
        vmin, vmax = 0, 1000

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
                     map_style=pdk.map_styles.ROAD, tooltip = tooltip), vmin, vmax

gdf = load_gdf()
gdf = filt_df(gdf, region_radio)
view_p = find_best_view(gdf, region_radio)
my_map, vmin, vmax = get_map(gdf, view_p[0], view_p[1], view_p[2],
    region_radio, time_radio, type_radio)

with row1_2:
    st.write(my_map)
with row1_3:
    st.pyplot(plot_color_gradients('plasma', vmin, vmax))

@st.cache(persist=True, allow_output_mutation=True)
def altris_compare_by_tot_miles_test(df, x_col, y_col, facet_col='city_name', use_log=True):
    df = df[[x_col, y_col, facet_col]]
    x_str = "{}:Q".format(x_col)
    y_str = "sum({}):Q".format(y_col)
    f_str = "{}:N".format(facet_col)
    my_bin = alt.Bin(extent=[1.0, 3.5], step=0.02) # alt.Bin(maxbins=100)
    x_axis = alt.X(x_str, bin=my_bin, scale=alt.Scale(type='log'))
    if use_log is False:
        x_axis = alt.X(x_str, bin=my_bin)
    return alt.Chart(df).mark_bar(opacity=0.8,
    ).encode(
        x=x_axis,
        y=alt.Y(y_str, stack=None, title="Total miles"),
        color=alt.Color(f_str, sort=city_order, title="city names"),
        tooltip=[alt.Tooltip(y_str, format='.2f')],
        row=alt.Row(f_str, sort=city_order)
    ).properties(height=50, width=800).interactive()

with row2_2:
    df1 = pd.DataFrame(gdf)
    city_order = ['Knoxville', 'Chattanooga', 'Nashville', 'Memphis']
    prefix_n = get_prefix_by_time(time_radio)
    suffix_n = get_suffix_by_type(type_radio)
    col_name = prefix_n + suffix_n

    line = altris_compare_by_tot_miles_test(df1, col_name, 'Miles')
    if suffix_n == "_ed":
        line = altris_compare_by_tot_miles_test(df1, col_name, 'Miles', use_log=False)

    st.altair_chart(line, use_container_width=True)




