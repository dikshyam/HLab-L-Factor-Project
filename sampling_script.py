
# Usage: python3 sampling_script.py --tables_json tables.json --sample_size 1500 --word_count_threshold 500


import subprocess
import logging
import os
import json
import sys
import argparse
import getpass
import MySQLdb
import numpy as np
import pandas as pd
import shutil
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

def get_connection(data_args):
    myDB = URL.create(drivername='mysql', host=data_args['hostname'],
                      database=data_args['db'], query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})
    engine = create_engine(myDB, encoding='latin1')
    conn = engine.connect()
    return conn

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def main(tables_json, sample_size, word_count_threshold):
    # Get current date
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Load tables from JSON file
    with open(tables_json, 'r') as f:
        tables = json.load(f)
    
    # Set working directory and database name
    working_dir = "/home/dmohanty/L-Factor/"
    db = "LFactor2023"
    
    data_args = {
        'db': db,
        'hostname': 'localhost'
    }

    data_args = dotdict(data_args)

    # Initialize an empty list to store table data
    table_data = []

    for key, value in tables.items():
        n_users = sample_size
        print(value)
        db = "LFactor2023"
        user = getpass.getuser()
        conn = MySQLdb.connect(read_default_file="~/.my.cnf", db=db, user=user)
        
        # Load the table data
        df = pd.read_sql_query(sql=f"SELECT * FROM {value['original_table']}", con=conn)
        conn.close()

        df[value['original_userid']] = df[value['original_userid']].astype(str)
        # Initial row count
        initial_row_count = df.shape[0]
        
        # Filter out rows with missing user IDs
        df = df[~df[value['original_userid']].isna()]

        # Calculate word count
        df['word_count'] = df[value['original_message_field']].apply(lambda x: len(x.split()))
        user_word_count = df.groupby(value['original_userid'])['word_count'].sum().reset_index()

        # Filter users with word count < word_count_threshold
        wc = word_count_threshold
        filtered_users_before_sampling = user_word_count[user_word_count['word_count'] > wc]
        
        if filtered_users_before_sampling.shape[0] < sample_size:
            wc = 50
            filtered_users_before_sampling = user_word_count[user_word_count['word_count'] > wc]
        if n_users > filtered_users_before_sampling.shape[0]:
            n_users = filtered_users_before_sampling.shape[0]
        filtered_users_after_sampling = filtered_users_before_sampling.sample(n_users, random_state=42)
        
        print(f"Number of Users with {wc} threshold: {filtered_users_after_sampling.shape[0]}")

        new_table_name = value['original_table'] + "_filtered_usergroups"
        filtered_users_after_sampling[value['original_userid']] = filtered_users_after_sampling[value['original_userid']].astype(str)

        # Filter the original DataFrame based on the sampled users
        filtered_df = df[df[value['original_userid']].isin(filtered_users_after_sampling[value['original_userid']])]
        final_row_count = filtered_df.shape[0]
        
        # Calculate average word count per user in the filtered data
        avg_word_count_per_user = filtered_users_after_sampling['word_count'].mean()
        
        # Calculate average message count per user in the filtered data
        avg_message_count_per_user = filtered_df.groupby(value['original_userid']).size().mean()
        
        # Calculate average message size in the filtered data
        avg_message_size = filtered_df['word_count'].mean()
        
        # Save the filtered users to a new table
        conn = get_connection(data_args)
        filtered_users_after_sampling.to_sql(name=new_table_name, con=conn, index=False, if_exists="replace")
        print(f"Writing data to db: {filtered_users_after_sampling.shape[0]}")
        conn.close()
        
        # Append the table's data to the list
        table_data.append({
            'table_name': value['original_table'],
            'initial_row_count': initial_row_count,
            'number_of_users': user_word_count.shape[0],
            'word_count_threshold': wc,
            'filtered_users_before_sampling': filtered_users_before_sampling.shape[0],
            'filtered_users_after_sampling': filtered_users_after_sampling.shape[0],
            'final_row_count': final_row_count,
            'avg_word_count_per_user': round(avg_word_count_per_user, 2),
            'avg_message_count_per_user': round(avg_message_count_per_user, 2),
            'avg_message_size': round(avg_message_size, 2)
        })

    # Convert the list to a DataFrame
    summary_df = pd.DataFrame(table_data)

    # Display the summary DataFrame
    print(summary_df)


    # Save the summary DataFrame to a CSV file
    summary_df.to_csv(f"summary_stats_{current_date}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample data from tables and generate summary statistics.')
    parser.add_argument('--tables_json', type=str, required=True, help='Path to JSON file containing tables dictionary')
    parser.add_argument('--sample_size', type=int, required=True, help='Number of users to sample')
    parser.add_argument('--word_count_threshold', type=int, required=True, help='Minimum word count threshold for user inclusion')
    
    args = parser.parse_args()
    
    main(args.tables_json, args.sample_size, args.word_count_threshold)
