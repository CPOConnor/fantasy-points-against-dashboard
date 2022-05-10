#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import os
import urllib.request
from PIL import Image
import json
import pickle
from time import sleep

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 400)
pd.set_option('display.max_colwidth', 500)  # allow relatively long strings for desc column


def refresh_pickles():
    years = [2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006]
    for year in years:
        print(f'Starting {year} (getting data/pickle)')
        get_fpa_by_position(year)
        print(f'Finished {year} (getting data/pickle)')

def get_pbp_data(year):
    path = f'cache/play_by_play_{year}.csv.gz'
    if os.path.isfile(path):
        return pd.read_csv(path, compression='gzip', low_memory=False)
    else:
        data = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_' + str(year) + '.csv.gz?raw=True',
                           compression='gzip', low_memory=False)
        # Filter before saving to lower storage cost
        data = data.loc[(data.special_teams_play == 0)
                        & (data.epa.isna() == False)
                        & (data.season_type == 'REG')
                        & (data.play_type.isin(['pass', 'run']))]
        # # Commenting out saving, I would prefer not to do this on the heroku app
        # # We should be sourcing from pickles anyway.
        # data.to_csv(f'cache/play_by_play_{year}.csv.gz', compression='gzip', index=False)
        return data


def get_rushers(df):
    rushers = df.loc[(df.rusher_player_id.isna() == False)].groupby([df.rusher_player_name,
                                                                     df.rusher_player_id,
                                                                     df.week,
                                                                     df.posteam,
                                                                     df.defteam], dropna=False) \
        .count()[['play_id']].reset_index()
    rushers.rename(columns={rushers.columns[0]: 'player',
                            rushers.columns[1]: 'player_id',
                            rushers.columns[5]: 'plays'}, inplace=True)
    return rushers


def get_passers(df):
    passers = df.loc[(df.passer_player_id.isna() == False)].groupby([df.passer_player_name,
                                                                     df.passer_player_id,
                                                                     df.week,
                                                                     df.posteam,
                                                                     df.defteam], dropna=False) \
        .count()[['play_id']].reset_index()
    passers.rename(columns={passers.columns[0]: 'player',
                            passers.columns[1]: 'player_id',
                            passers.columns[5]: 'plays'}, inplace=True)
    return passers


def get_receivers(df):
    receivers = df.loc[(df.receiver_player_id.isna() == False)].groupby([df.receiver_player_name,
                                                                         df.receiver_player_id,
                                                                         df.week,
                                                                         df.posteam,
                                                                         df.defteam], dropna=False) \
        .count()[['play_id']].reset_index()
    receivers.rename(columns={receivers.columns[0]: 'player',
                              receivers.columns[1]: 'player_id',
                              receivers.columns[5]: 'plays'}, inplace=True)
    return receivers


def get_players(data):
    rushers = get_rushers(data)
    passers = get_passers(data)
    receivers = get_receivers(data)

    players = pd.concat([rushers, passers, receivers], ignore_index=True)

    return players.groupby(['player',
                            'player_id',
                            'week',
                            'posteam',
                            'defteam'], dropna=False).sum()[['plays']].reset_index()


def get_roster_data(year):
    # look for cached results first
    path = f'cache/roster_{year}.csv.gz'
    if os.path.isfile(path):
        return pd.read_csv(path, compression='gzip', low_memory=False)
    else:
        data = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_'
                           + str(year) + '.csv', low_memory=False)
        # Avoid saving to heroku app
        #data.to_csv(path, compression='gzip', index=False)
        return data


def get_player_stats_data(year):
    # look for cached results first
    path = f'cache/player_stats_{year}.csv.gz'
    if os.path.isfile(path):
        return pd.read_csv(path, compression='gzip', low_memory=False)
    else:
        data = pd.read_csv('https://github.com/nflverse/nflfastR-data/blob/master/data/' \
                           f'player_stats/player_stats_{year}.csv.gz?raw=True',
                           compression='gzip', low_memory=False)
        # Avoid saving to heroku app
        #data.to_csv(path, compression='gzip', index=False)
        return data


def get_two_year_roster_data(year):
    roster_data = get_roster_data(year)
    # attempt to get roster_data from prev year
    roster_data_prev_year = get_roster_data(year - 1)

    # combine roster data with previous year roster data
    two_year_roster_data = pd.concat([roster_data, roster_data_prev_year])

    # get indexes for max year rows 
    idx = two_year_roster_data.groupby(['gsis_id'])['season'].transform(max) == two_year_roster_data['season']

    # subset down to max year rows (removes duplicates for most players)
    two_year_roster_data = two_year_roster_data[idx]

    return two_year_roster_data


def get_fpa_by_position(year):
    cache_path = f'cache/fpa_by_positon_{year}.pkl'
    fpa_by_position = {}
    if os.path.exists(cache_path):
        fpa_by_position = pd.read_pickle(cache_path)
    else:
        data = get_pbp_data(year)
        players = get_players(data)
        player_stats_data = get_player_stats_data(year)
        two_year_roster_data = get_two_year_roster_data(year)

        joined_data = pd.merge(players,  # left
                               player_stats_data,  # right
                               how='inner',
                               left_on=['player_id', 'week'],
                               right_on=['player_id', 'week'])

        tri_data = pd.merge(joined_data,  # left
                            two_year_roster_data,  # right
                            how='left',
                            left_on=['player_id'],
                            right_on=['gsis_id'])

        unconventional_ball_carriers = tri_data.loc[(tri_data.position.isin(['QB', 'RB', 'FB', 'TE', 'WR']) == False)]

        non_skill_pos_data = unconventional_ball_carriers.groupby(['week', 'posteam', 'defteam'], dropna=False).sum()[
            ['fantasy_points']]
        non_skill_pos_data.reset_index(inplace=True)
        non_skill_pos_data = non_skill_pos_data.loc[(non_skill_pos_data.fantasy_points > 6)]
        # create a position column to represent these rows when joined
        non_skill_pos_data['position'] = 'non_skill'

        fixed_data = pd.concat([tri_data, non_skill_pos_data], ignore_index=True)

        fixed_data.loc['position'] = fixed_data['position'].replace('FB', 'RB')
        fixed_data = fixed_data.loc[(fixed_data.position.isin(['QB', 'RB', 'WR', 'TE', 'non_skill']))]

        fpa_by_position = fixed_data.groupby(['week', 'defteam', 'posteam', 'position'], dropna=False).sum()[
            ['fantasy_points', 'fantasy_points_ppr']]
        fpa_by_position.reset_index(inplace=True)
        # save data in pickle
        fpa_by_position.to_pickle(cache_path)

    return fpa_by_position


def get_logo_dict():
    # Check for existing images
    image_dir = os.getcwd() + '/images/team_logos/'
    urls = pd.read_csv('https://raw.githubusercontent.com/statsbylopez/BlogPosts/master/nfl_teamlogos.csv')
    logos = os.listdir(os.getcwd() + '/images/team_logos')
    # For now I'm just going to check we have 32 team logos.
    # could refactor this to pass team, but I'm not sure how it works with
    # teams that switch names
    if len(logos) < 32:
        # some/all images missing, get from source
        for i in range(0, len(urls)):
            urllib.request.urlretrieve(urls['url'].iloc[i], image_dir + urls['team_code'].iloc[i] + '.png')
            sleep(.3)  # avoid too many requests

        logos = os.listdir(os.getcwd() + '/images/team_logos')

    logo_dict = {}

    for i in logos:
        team_name = i.split('.')[0]
        path = os.getcwd() + '/images/team_logos/' + str(i)
        logo_dict[team_name] = path

    return logo_dict


# Gets data shared by both barchart and linechart
def get_fpa_data(year, pos, include_final_week):
    # Get dataset
    fpa_by_position = get_fpa_by_position(year)

    # include_final_week comes in as ['Include Week 18'], ['Include Week 17'], or [] so convert to bool
    bool_inc_final_week = True if include_final_week else False

    # get max week (17 or 18)
    weeks = (fpa_by_position['week'].unique())
    max_week = int(weeks.max())

    # filter on parameters
    fpa_data = fpa_by_position.loc[
        (fpa_by_position.position == pos) & (bool_inc_final_week | (fpa_by_position.week != max_week))].copy()

    # add opponent averages
    # make dictionary with keys = teams (offesnse aka 'posteam') and values = average fpoints
    opp_avg_d = fpa_data.groupby(['posteam']).mean()[['fantasy_points', 'fantasy_points_ppr']].T.to_dict()

    for index_label, row_series in fpa_data.iterrows():
        fpa_data.at[index_label, 'opp_avg'] = opp_avg_d[row_series['posteam']]['fantasy_points']
        fpa_data.at[index_label, 'opp_avg_ppr'] = opp_avg_d[row_series['posteam']]['fantasy_points_ppr']

    return fpa_data


def get_def_avg_barchart_data(year, pos, scoring_system, include_final_week, graph_type):
    fpa_data = get_fpa_data(year, pos, include_final_week)

    barchart_data = fpa_data.groupby(['defteam']).mean()[
        ['fantasy_points', 'fantasy_points_ppr', 'opp_avg', 'opp_avg_ppr']].reset_index()

    # add half ppr col 
    barchart_data['fantasy_points_half_ppr'] = (barchart_data['fantasy_points'] + barchart_data[
        'fantasy_points_ppr']) / 2

    # add opp avg / diff cols 
    barchart_data['opp_avg_half_ppr'] = (barchart_data['opp_avg'] + barchart_data['opp_avg_ppr']) / 2

    barchart_data['diff_opp_avg'] = (barchart_data['fantasy_points'] - barchart_data['opp_avg'])
    barchart_data['diff_opp_avg_half_ppr'] = (
            barchart_data['fantasy_points_half_ppr'] - barchart_data['opp_avg_half_ppr'])
    barchart_data['diff_opp_avg_ppr'] = (barchart_data['fantasy_points_ppr'] - barchart_data['opp_avg_ppr'])

    # find the overall league average
    league_avg = barchart_data[['fantasy_points']].mean()

    # get the number of teams
    len_team = len(barchart_data)

    # create array of league_avg as long as the number of teams to
    # create derived column easily and set up baseline in bar graph
    arr_league_avg = ([league_avg.fantasy_points] * len_team)

    # create derived column to show difference between their fpa and league average
    barchart_data['difference'] = barchart_data['fantasy_points'] - arr_league_avg

    # create new column with league average to allow access in hover
    barchart_data['league_average'] = arr_league_avg

    # add league average for other scoring systems
    league_avg_half = barchart_data[['fantasy_points_half_ppr']].mean()
    arr_league_avg_half = ([league_avg_half.fantasy_points_half_ppr] * len_team)
    barchart_data['league_average_half_ppr'] = arr_league_avg_half

    league_avg_ppr = barchart_data[['fantasy_points_ppr']].mean()
    arr_league_avg_ppr = ([league_avg_ppr.fantasy_points_ppr] * len_team)
    barchart_data['league_average_ppr'] = arr_league_avg_ppr

    # add difference for other scoring systems
    barchart_data['difference_half_ppr'] = barchart_data['fantasy_points_half_ppr'] - arr_league_avg_half
    barchart_data['difference_ppr'] = barchart_data['fantasy_points_ppr'] - arr_league_avg_ppr

    if scoring_system == 'Standard':
        if graph_type == 'Raw':
            barchart_data.sort_values('fantasy_points', inplace=True)
        else:
            barchart_data.sort_values('diff_opp_avg', inplace=True)
    elif scoring_system == 'Half-PPR':
        if graph_type == 'Raw':
            barchart_data.sort_values('fantasy_points_half_ppr', inplace=True)
        else:
            barchart_data.sort_values('diff_opp_avg_half_ppr', inplace=True)
    elif scoring_system == 'PPR':
        if graph_type == 'Raw':
            barchart_data.sort_values('fantasy_points_ppr', inplace=True)
        else:
            barchart_data.sort_values('diff_opp_avg_ppr', inplace=True)
    return barchart_data


def get_weekly_linechart_data(year, pos, scoring_system, include_final_week, graph_type, team):
    fpa_data = get_fpa_data(year, pos, include_final_week)

    fpa_data = fpa_data.loc[(fpa_data.defteam == team)]

    fpa_data['fantasy_points_half_ppr'] = (fpa_data['fantasy_points'] + fpa_data['fantasy_points_ppr']) / 2
    fpa_data['opp_avg_half_ppr'] = (fpa_data['opp_avg_ppr'] + fpa_data['opp_avg']) / 2

    return fpa_data


def get_final_week(year):
    # Get dataset
    fpa_by_position = get_fpa_by_position(year)

    # get max week (17 or 18)
    weeks = (fpa_by_position['week'].unique())
    return int(weeks.max())


# define get hover dictionary 
def get_hover_dict(scoring_system):
    h_d = {'fantasy_points': scoring_system == "Standard",
           'fantasy_points_ppr': scoring_system == "PPR",
           'fantasy_points_half_ppr': scoring_system == "Half-PPR",
           'league_average': scoring_system == "Standard",
           'league_average_ppr': scoring_system == "PPR",
           'league_average_half_ppr': scoring_system == "Half-PPR",
           'difference': False,
           'difference_half_ppr': False,
           'difference_ppr': False}

    return h_d


def get_rank_dict(df, scoring_system):
    # Create a dictionary of each teams season rank to determine place on y axis
    d_rank = {}
    if scoring_system == 'Standard':
        df_grouped = df.groupby(['defteam']).mean()[['fantasy_points']].reset_index()
        df_grouped.sort_values('fantasy_points', inplace=True)
    elif scoring_system == 'Half-PPR':
        df_grouped = df.groupby(['defteam']).mean()[['fantasy_points_half_ppr']].reset_index()
        df_grouped.sort_values('fantasy_points_half_ppr', inplace=True)
    elif scoring_system == 'PPR':
        df_grouped = df.groupby(['defteam']).mean()[['fantasy_points_ppr']].reset_index()
        df_grouped.sort_values('fantasy_points_ppr', inplace=True)

    for i in range(len(df_grouped.defteam)):
        rank = i
        team = df.iloc[i].defteam
        d_rank[team] = rank

    return d_rank


def get_teams(year):
    return get_fpa_by_position(year)['defteam'].unique()


def get_names(pos):
    return {'opp_avg': f'Opponent {pos} Average Points',
            'fantasy_points': 'Fantasy Points (Std.)',
            'fantasy_points_half_ppr': 'Fantasy Points (Half-PPR)',
            'fantasy_points_ppr': 'Fantasy Points (PPR)',
            'opp_avg_ppr': f'Opponent {pos} Average Points',
            'opp_avg_half_ppr': f'Opponent {pos} Average Points',
            'difference': 'Difference from League Average',
            'difference_ppr': 'Difference from League Average',
            'difference_half_ppr': 'Difference from League Average',
            'diff_opp_avg': 'Difference from Opponent Average',
            'diff_opp_avg_ppr': 'Difference from Opponent Average',
            'diff_opp_avg_half_ppr': 'Difference from Opponent Average',
            '': ''
            }


### Layout

# stylesheet with the .dbc class
dbc_css = (
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.4/dbc.min.css"
)

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc_css],
           meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, '
                                                      'minimum-scale=0.5,'}])
server = app.server
# define defaults
default_pos = 'WR'
default_scor = 'Half-PPR'
default_graph_type = 'Raw'
default_year = 2021

# Define components
header = html.H4(f'{default_year} Fantasy Points Against - {default_pos}', id='barchart-title',
                 className="bg-primary text-white p-2 mb-2 text-center")

radio_pos = dcc.RadioItems(
    id='radio-position',
    options=['WR', 'RB', 'TE', 'QB'],
    labelStyle={'display': 'block'},
    inputClassName='options-input',
    value=default_pos
)

radio_scoring_system = dcc.RadioItems(
    id='radio-scoring-system',
    options=['Standard', 'Half-PPR', 'PPR'],
    labelStyle={'display': 'block'},
    inputClassName='options-input',
    value=default_scor
)

ck_box_include_last_week = dcc.Checklist(
    id='ck-box-include-final-week',
    inline=True,
    options=['Include Week ' + str(get_final_week(default_year))],
    inputClassName='options-input',
    value=[]
)

radio_graph_type = dcc.RadioItems(
    id='radio-graph-type',
    options=['Raw', 'Strength of schedule adjusted'],
    labelStyle={'display': 'block'},
    inputClassName='options-input',
    value=default_graph_type
)

dropdown_year = dcc.Dropdown(
    id='dropdown-year',
    options=[2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006],
    value=default_year
)

controls = dbc.Card([
    html.H5("Options", className="card-title"),
    html.Hr(),
    dbc.Row(
        [
            dbc.Col([html.Label('Year', className='options-label'), dropdown_year], width=12,
                    md={'size': 3, 'offset': 0}),
            dbc.Col([html.Label('Position', className='options-label'), radio_pos], width=12, md=2),
            dbc.Col([html.Label('Format', className='options-label'), radio_scoring_system], width=12, md=2),
            dbc.Col([html.Label('Type', className='options-label'), radio_graph_type], width=12, md=2),
            dbc.Col([html.Label('Final week', className='options-label'), ck_box_include_last_week], width=12, md=3)
        ])],
    body=True)

# dropdown_year,radio_pos, radio_scoring_system, ck_box_include_last_week, radio_graph_type],) #remove

# Wrap graphs in loading spinners
loading1 = dbc.Spinner(
    size="lg",
    fullscreen=False,
    color='primary',
    children=dcc.Graph(id='barchart-season')
)
loading2 = dbc.Spinner(
    size="lg",
    fullscreen=False,
    color='primary',
    children=dcc.Graph(id='linechart-weekly')
)
tab1 = dbc.Tab([loading1], label='All Teams', tab_id='tab_1')
team_dropdown = dcc.Dropdown(id='dropdown-team', options=get_teams(default_year))
tab2 = dbc.Tab([dbc.Row(dbc.Col([team_dropdown], width=12, md=3)), loading2],
               label='Single team',
               tab_id='tab_2')
tabs = dbc.Tabs([tab1, tab2], id='tabs', active_tab='tab_1')
footer = dbc.Row(
                dbc.Col(html.Div(children='Chart: @CocoChart Data: nflverse', id='footer-div'))
)
app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                dbc.Col([controls])
            ]
        ),
        dbc.Row(
            [
                dbc.Col(tabs)
            ]
        ),
        footer
    ],
    fluid=True,
    className="dbc",
)


### Callbacks

@app.callback(
    Output(component_id='barchart-season', component_property='figure'),
    Input(component_id='dropdown-year', component_property='value'),
    Input(component_id='radio-position', component_property='value'),
    Input(component_id='radio-scoring-system', component_property='value'),
    Input(component_id='ck-box-include-final-week', component_property='value'),
    Input(component_id='radio-graph-type', component_property='value'))
def update_barchart_season(year, pos, scoring_system, include_final_week, graph_type):
    df = get_def_avg_barchart_data(year, pos, scoring_system, include_final_week, graph_type)
    h_d = get_hover_dict(scoring_system)
    fig = {}
    scale = 'RdYlGn'

    if scoring_system == 'Standard':
        if graph_type == 'Raw':
            fig = px.bar(
                df, y='difference', x='defteam',
                base='league_average', hover_data=h_d, color='difference', color_continuous_scale=scale)
        else:
            fig = px.bar(
                df, y='diff_opp_avg', x='defteam', hover_data=h_d, color='diff_opp_avg', color_continuous_scale=scale)
    elif scoring_system == 'Half-PPR':
        if graph_type == 'Raw':
            fig = px.bar(
                df, y='difference_half_ppr', x='defteam',
                base='league_average_half_ppr', hover_data=h_d, color='difference_half_ppr',
                color_continuous_scale=scale)
        else:
            fig = px.bar(
                df, y='diff_opp_avg_half_ppr', x='defteam', hover_data=h_d, color='diff_opp_avg_half_ppr',
                color_continuous_scale=scale)

    elif scoring_system == 'PPR':
        if graph_type == 'Raw':
            fig = px.bar(
                df, y='difference_ppr', x='defteam',
                base='league_average_ppr', hover_data=h_d, color='difference_ppr', color_continuous_scale=scale)
        else:
            fig = px.bar(
                df, y='diff_opp_avg_ppr', x='defteam', hover_data=h_d, color='diff_opp_avg_ppr',
                color_continuous_scale=scale)

    fig.update_xaxes(
        tickmode='linear',
        showticklabels=False,
        title=None,
        showgrid=False
    )
    append_title = ""
    if graph_type != "Raw":
        append_title = " - Opponent Avg Points"
    fig.update_yaxes(
        title=f'Fantasy Points Against {pos}' + append_title,
        showgrid=False
    )

    # make hover template from hover dict
    customdata = {}
    if graph_type == "Raw":

        if scoring_system == "Standard":
            customdata = df[['fantasy_points', 'league_average']]
        elif scoring_system == "Half-PPR":
            customdata = df[['fantasy_points_half_ppr', 'league_average_half_ppr']]
        elif scoring_system == "PPR":
            customdata = df[['fantasy_points_ppr', 'league_average_ppr']]

        hover_templates = ['Team: %{x}',
                           'Average Points Against: %{customdata[0]:.2f}',
                           'League Average: %{customdata[1]:.2f}',
                           'Click for more info']
    else:

        if scoring_system == "Standard":
            customdata = df[['fantasy_points', 'opp_avg', 'diff_opp_avg']]
        elif scoring_system == "Half-PPR":
            customdata = df[['fantasy_points_half_ppr', 'opp_avg_half_ppr', 'diff_opp_avg_half_ppr']]
        elif scoring_system == "PPR":
            customdata = df[['fantasy_points_ppr', 'opp_avg_ppr', 'diff_opp_avg_ppr']]

        hover_templates = ['Team: %{x}',
                           'Average Points Against: %{customdata[0]:.2f}',
                           'Opponent Average Points:  %{customdata[1]:.2f}',
                           'Difference: %{customdata[2]:.2f}',
                           'Click for more info']

    fig.update_traces(customdata=customdata, hovertemplate="<br>".join(hover_templates))

    # add images
    baseline = 0
    margin = .5  # defines margin between logo and bar
    d_rank = get_rank_dict(df, scoring_system)

    if graph_type != 'Raw':
        baseline = 0
    else:
        if scoring_system == "Standard":
            baseline = df.iloc[0].league_average
        elif scoring_system == "Half-PPR":
            baseline = df.iloc[0].league_average_half_ppr
        elif scoring_system == "PPR":
            baseline = df.iloc[0].league_average_ppr

    logo_dict = get_logo_dict()

    for i in range(len(df)):
        if scoring_system == "Standard":
            if graph_type == "Raw":
                base_mod = baseline - margin if df.iloc[i].difference >= 0 else baseline + margin
            else:
                base_mod = baseline - margin if df.iloc[i].diff_opp_avg >= 0 else baseline + margin
        elif scoring_system == "Half-PPR":
            if graph_type == "Raw":
                base_mod = baseline - margin if df.iloc[i].difference_half_ppr >= 0 else baseline + margin
            else:
                base_mod = baseline - margin if df.iloc[i].diff_opp_avg_half_ppr >= 0 else baseline + margin

        elif scoring_system == "PPR":
            if graph_type == "Raw":
                base_mod = baseline - margin if df.iloc[i].difference_ppr >= 0 else baseline + margin
            else:
                base_mod = baseline - margin if df.iloc[i].diff_opp_avg_ppr >= 0 else baseline + margin

        defteam = df.iloc[i].defteam
        fig.add_layout_image(
            source=Image.open(logo_dict[defteam]),
            xref="x",
            yref="y",
            y=base_mod,
            x=d_rank[defteam],
            xanchor="center",
            yanchor="middle",
            sizex=1,
            sizey=1, )

    fig.update_coloraxes(showscale=False)
    fig.update_layout({'plot_bgcolor' : 'rgba(0,0,0,0)'})
    return fig


@app.callback(
    Output('linechart-weekly', 'figure'),
    Input('dropdown-year', 'value'),
    Input('radio-position', 'value'),
    Input('radio-scoring-system', 'value'),
    Input('ck-box-include-final-week', 'value'),
    Input('radio-graph-type', 'value'),
    Input('dropdown-team', 'value'))
def update_weekly_components(year, pos, scoring_system, include_final_week, graph_type, team):
    if not team:
        return {}
    weekly_data = get_weekly_linechart_data(year, pos, scoring_system, include_final_week, graph_type, team)
    fig = {}
    y0 = 'fantasy_points'
    y1 = 'opp_avg'
    if scoring_system == 'Half-PPR':
        y0 = 'fantasy_points_half_ppr'
        y1 = 'opp_avg_half_ppr'
    elif scoring_system == 'PPR':
        y0 = 'fantasy_points_ppr'
        y1 = 'opp_avg_ppr'

    fig = px.line(weekly_data, x='week', y=[y0, y1], title=f'{team} Fantasy Points Against {pos} By Week')

    customdata = weekly_data[['posteam', y1, y0]]
    hover_templates = ['Week: %{x}',
                       'Opp: %{customdata[0]}',
                       'Fantasy Points: %{customdata[2]:.2f}',
                       '%{customdata[0]} ' + pos + ' Avg ' + 'Points: %{customdata[1]:.2f}']

    fig.update_traces(customdata=customdata, hovertemplate="<br>".join(hover_templates))
    # Create a name dictionary to show readable names of columns
    name_d = get_names(pos)
    fig.for_each_trace(lambda t: t.update(name=name_d[t.name]))
    fig.update_yaxes(title='Fantasy Points')
    return fig


@app.callback(
    Output('dropdown-team', 'value'),
    Input('barchart-season', 'clickData'))
def update_selected_team(click_data):
    if click_data:
        return click_data['points'][0]['x']
    else:
        return None


@app.callback(
    Output(component_id='barchart-title', component_property='children'),
    Input(component_id='dropdown-year', component_property='value'),
    Input(component_id='radio-position', component_property='value'),
    Input(component_id='radio-scoring-system', component_property='value'))
def update_title(year, pos, scoring_system):
    return f'{year} {scoring_system} Fantasy Points Against - {pos}'


@app.callback(
    Output('tabs', 'active_tab'),
    Input('barchart-season', 'clickData'))
def select_single_team_tab(click_data):
    if click_data:
        return 'tab_2'
    else:
        return 'tab_1'


@app.callback(
    Output('dropdown-teams', 'options'),
    Input('dropdown-year', 'value'))
def update_teams(year):
    return get_teams(year)


@app.callback(
    Output('ck-box-include-final-week', 'options'),
    Input('dropdown-year', 'value'))
def update_checkbox_include_final_week(year):
    return ['Include Week ' + str(get_final_week(year))]


if __name__ == '__main__':
    #refresh_pickles()
    app.run_server(debug=False, port=8050)
