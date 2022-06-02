import pandas as pd
import datetime as dt
import regex as re
import os
from pathlib import Path

def get_all_data(dirpath = Path('raw_data/')):
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
    return df

def get_df(list_of_csvs= ['SPY_data.csv','TR_data.csv','VIX_data.csv']):
    """ Concat a list of csvs into a Single df"""
    list_dfs = [pd.read_csv(f'raw_data/{_csv}', parse_dates = True, infer_datetime_format = True) for _csv in list_of_csvs]
    for df in list_dfs:
        df.drop_duplicates(inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['date'] = pd.to_datetime(df['date'])
        df = drop_unnamed(df)
        df.set_index('date',inplace=True)
    clean_headers = clean_col_names(list_dfs,list_of_csvs)
    merged_df = pd.concat(list_dfs, axis=1, join= 'inner')  
    merged_df.columns = clean_headers
    return merged_df

