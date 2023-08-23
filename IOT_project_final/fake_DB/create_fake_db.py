import csv
import numpy as np
import pandas as pd

import sqlite3
from sqlite3 import Error, sqlite_version

import signal
import sys

DATA_FILE = 'fake_db.csv'
database = r"pythonsqlite.db"

def handler(signumber, frame):
    if connection:
        connection.close()
    sys.exit("Program Interrupted.")
            

def create_connection(db_file):
    """Create a connection to SQLite database"""
    connection = None
    try:
        connection = sqlite3.connect(db_file, check_same_thread=False)
    except Error as err:
        print(f"Error during connection to database as :'{err}'")
    
    return connection

def create_table(connection, create_table_query):
    try:
        cursor = connection.cursor()
        cursor.execute(create_table_query)
    except Error as err:
        print(f"Error during creation of table as :'{err}'")
    return cursor

def insert_values(cursor, data): # add connection.commit() after calling this function
    assert len(data) == 5
    insert_value_query = f"INSERT INTO predictions VALUES('{data[0]}', '{data[1]}', '{data[2]}', '{data[3]}', '{data[4]}')"
    try:
        cursor.execute(insert_value_query)
    except Error as err:
        print(f"Error during insertion as :'{err}'")

def select_function(cursor, select_query):
    try:
        cursor.execute(select_query)
        rows = cursor.fetchall()
        for row in rows:
            print(row)
    except Error as err:
        print(f"Error in selecting as : '{err}'")

def main():
    with open(DATA_FILE, 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=","))
    
    for row in data[1:]:
        for i in range(len(row)):
            row[i] = float(row[i])
    
    data.remove(data[0])
   
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]= int(data[i][j])
        dataToInsert = data[i]
        insert_values(cursor, dataToInsert)
        connection.commit()  
    
    query = "SELECT * FROM predictions;"
    select_function(cursor, query)

    while True:
        pass
        signal.signal(signal.SIGINT, handler)

if __name__ == '__main__':
   connection = create_connection(database)
   connection.cursor().execute("DROP TABLE IF EXISTS predictions")
   create_statement = "CREATE TABLE IF NOT EXISTS predictions(id INTEGER, timestp INTEGER, duration INTEGER, day VARCHAR, hour VARCHAR)"
   cursor = create_table(connection, create_statement)
   main()
