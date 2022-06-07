import pandas as pd
import datetime as dt
import regex as re
import os
from pathlib import Path

def get_all_raw_data(dirpath = Path('raw_data/')):
    "returns a list of csvs from a folder"
    csv_list = [filename for filename in os.listdir(dirpath)]
    return csv_list

def clean_col_names(list_dfs, list_csvs):
    """ Appends the data source name to the header of a column """
    list_cols = [df.columns.to_list() for df in list_dfs]
    zipped = zip(list_csvs,list_cols)
    name_list = []
    for tup in list(zipped):
        name = tup[0]
        match = re.match(r'(\w+)_',name)
        name = match.group(1)
        for col in tup[1]:
            if name.lower() != col.lower():
                name_list.append((name.upper()+'_'+col.lower()))
            else:
                name_list.append(name.upper())
    return name_list

def drop_unnamed(df):
    """ Drops columns containing the string 'unnamed' """
    for col in df.columns:
        if 'unnamed' in str(col):
            df.drop(columns= col, inplace=True)
    return None

def get_df(list_of_csvs= ['SPY_data.csv','TR_data.csv','VIX_data.csv']):
    """ Concat a list of csvs into a Single df"""
    list_dfs = [pd.read_csv(f'raw_data/{_csv}', parse_dates = True, infer_datetime_format = True) for _csv in list_of_csvs]
    for df in list_dfs:
        df.drop_duplicates(inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date',inplace=True)
    clean_headers = clean_col_names(list_dfs,list_of_csvs)
    merged_df = pd.concat(list_dfs, axis=1, join= 'inner')  
    merged_df.columns = clean_headers
    drop_unnamed(merged_df)
    return merged_df

def get_day_names(df):
    """Adds the day names for calc on weekly range"""
    df.reset_index(inplace=True)
    df['DayOfWeek'] = df['date'].dt.day_name()
    df.set_index('date', inplace=True)
    return df

def calc_weekly_range(df):
    """ Must run "get_day_names" first"""
    df = get_day_names(df)
    week_high = 0  
    week_low = 9999999
    for index, row in df.iterrows():
        if df.loc[index, 'DayOfWeek'] == 'Monday':
            week_high = df.loc[index,'SPY_high']
            week_low = df.loc[index,'SPY_low']
        else: 
            if df.loc[index,'SPY_high'] > week_high:
                week_high=df.loc[index,'SPY_high']
            if df.loc[index,'SPY_low'] < week_low:
                week_low=df.loc[index,'SPY_low']
            if df.loc[index,'DayOfWeek'] == 'Friday':
                df.loc[index,'weekly_range'] = week_high - week_low
    df = df.fillna(0)
    return df.drop(columns="DayOfWeek")

def grp_y_wk_d(df):
    """Must have datetime index"""
    return df.groupby(by = [df.index.isocalendar().year, 
    df.index.isocalendar().week,df.index.isocalendar().day]).mean()

def drop_off_weeks(df_input):
    idx = pd.IndexSlice
    df_input_ind = df_input.index.to_list()
    week_numbers = [tup[1] for tup in df_input_ind ]
    year_numbers = set([tup[0] for tup in df_input_ind ])
    year_numbers = sorted(list(year_numbers))
    df_copy = df_input.copy()

    for yr_num in year_numbers:
        for wk_num in week_numbers:
            
                try:
                    num_days = len(df_input.loc[idx[yr_num, wk_num]].index.to_list())
                    if num_days < 5:
                       df_copy.drop(labels = wk_num, level=1, axis=0, inplace= True)
                except:
                    continue
    df_copy = df_copy.reset_index()
    df_copy.drop(columns=['year', 'week', 'day'], inplace= True)
    return df_copy