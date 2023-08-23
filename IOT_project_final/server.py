import paho.mqtt.client as mqtt
import numpy as np
import json
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from PIL import Image
from os import listdir
from os.path import join
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import sqlite3
from sqlite3 import Error, sqlite_version
import signal
import sys

MODEL_NAME='nn_model.hd5'
model = load_model(MODEL_NAME)
dict = {0: "Machine off", 1: "Machine on"}
database = r"pythonsqlite.db"
POWERCONSUMPTION = 5400 # Watts
today = date = datetime.datetime.now().timestamp()
transformed_data = np.array([[0,today],[0,today],[0,today]]) # [prediction, date]

def handler(signumber, frame):
    if connection:
        connection.close()
    sys.exit("Program Interrupted.")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("IOT_nus_test")
    else:
        print("Connection failed with code: %d." %rc)

def classify(Id, data):
    date = datetime.datetime.now()
    print("Start classifying at: ", date)
    date = int(date.timestamp()) #int
    
    data=pd.DataFrame([data])
    #scaling values between 0 and 1
    with open(MODEL_NAME + '/extremes.csv', 'r') as csvfile:
        extreme_vals = list(csv.reader(csvfile, delimiter=";"))
    for row in extreme_vals[:]:
        for i in range(len(row)):
            row[i] = float(row[i])

    for i in range(len(extreme_vals)):
        data[i]= (data[i] - extreme_vals[i][0])/ (extreme_vals[i][1] - extreme_vals[i][0])
    
    #loading model
    result = model.predict(data)
    themax = np.argmax(result)

    return {"id": Id, "prediction": themax, "score":result[0][themax], "date": date}


def on_message(client, userdata, msg):
    print("—————————")
    if str(msg.payload.decode("utf-8")) == "requestData":
        print("Received message from website.")
        sendData(client)
    else:
        print("Received message from wemos.")
        handleData(msg.payload)


def sendData(client):
    """
    This function executes when the website sends a request for updated data from the server.
    The server sets up the data it will send, and sends it.
    """
    final = {}

    # First, gets the cycles per day.
    queryCycles = "select count(*) from predictions group by Day"
    cycles = select_function(cursor, queryCycles)
    for i in range(len(cycles)):
        cycles[i] = int(cycles[i][0])
    final["cyclesPerDay"] = cycles

    # Second, the average cycle duration.
    queryAvg = "select sum(duration) / count(duration) from predictions;"
    average = select_function(cursor, queryAvg)
    final["average"] = int(average[0][0] / 60)


    # Third, current states of the machines
    current = []
    for machine in transformed_data:
        value = []
        if machine[0]: ## The machine is working
            percentage = (datetime.datetime.now().timestamp() - machine[1]) / (final["average"]*60)
            print(percentage)
            if percentage >= 1:
                value.append(100)
                value.append(-1) # value to denote "available soon" on website
                
            else:
                value.append(int(percentage * 100))
                value.append(int((final["average"]*60 - (datetime.datetime.now().timestamp() - machine[1]))/60))
        else:
            value = [0,0]
        current.append(value)
    final["currState"] = current

    # Fourth, the energy.
    today = datetime.datetime.now().weekday()
    queryPower = "select sum(duration) from predictions where day = " + str(today) \
        + " and timestp >= " + str(datetime.datetime.now().timestamp() - 86400)
    totaltime = select_function(cursor, queryPower)

    if totaltime[0][0] is None:
        final["eneryToday"] = 0
    else:
        energy = 0.278 * POWERCONSUMPTION * totaltime[0][0] / 1000000 # Energy in Mega Joules
        final["eneryToday"] = energy

    # Fifth, gets the cycles per hour for current day
    queryHour = "select * from predictions where day = " + str(today)
    hours = select_function(cursor, queryHour)
    cyclesPerHour = []
    for i in range(24):
        cyclesPerHour.append(0)
        for j in range(len(hours)):
            if int(hours[j][4])==i:
                cyclesPerHour[i]+=1
                
    final["cyclesPerHour"] = cyclesPerHour

    # Finally, sending data to the website via MQTT.
    print("data sent to website:")
    print(final)
    client.publish("IOT_server_website_laundromates", json.dumps(final))


def handleData(msg):
    """
    This function executes when the message is received is data from the sensors sent by the WeMos device.
    It performs the prediction based on the data received and updates the database accordingly.
    """
    recv_dict = json.loads(msg)
    print("data received: ", recv_dict)
    
    data = np.array(recv_dict["data"])
    result = classify(recv_dict["id"], data)
    print("results of classification: ", result)
    
    result["id"] = int(result["id"])
    result["prediction"] = bool(result["prediction"])
    result["score"] = float(result["score"])
    result["date"] = float(result["date"])
    
    #conditions
    if transformed_data[result["id"]-1][0] != result["prediction"]:
        if(result["prediction"]): #turning on
            transformed_data[result["id"]-1][0]=result["prediction"]
            transformed_data[result["id"]-1][1]=result["date"]
        else: #turning off
            #sending data to database
            start = transformed_data[result["id"]-1][1]
            dataToInsert = [result["id"], start, result["date"]-start, datetime.datetime.now().weekday(), datetime.datetime.now().hour]
            insert_values(cursor, dataToInsert)
            connection.commit()
            print("===> data inserted to database: ", dataToInsert)
            
            transformed_data[result["id"]-1][0]=result["prediction"]
            transformed_data[result["id"]-1][1]=result["date"]  

def setup(hostname):
    client = mqtt.Client()
    client.on_connect= on_connect
    client.on_message = on_message
    client.connect(hostname, 1883, 60)
    client.loop_start()
    return client

#_______________________________________________________
#                 DATABASE FUNCTIONS

def create_connection(db_file):
    """Create a connection to SQLite database"""
    connection = None
    try:
        connection = sqlite3.connect(db_file, check_same_thread=False)
    except Error as err:
        print(f"Error during connection to database as :'{err}'")
    
    return connection

def create_table(connection, create_table_query):
    """Create a table from the create_table_sql statement
    :param conn: connection object
    :param create_table_query: CREATE TABLE query statement
    """
    try:
        cursor = connection.cursor()
        cursor.execute(create_table_query)
    except Error as err:
        print(f"Error during creation of table as :'{err}'")
    return cursor

def insert_values(cursor, data): #connection.commit() 
    """Insert values into the database
    :param cursor: the cursor object to execute queries
    :param insert_value_query the query to insert values into the table
    """
    assert len(data) == 5
    insert_value_query = f"INSERT INTO predictions VALUES('{data[0]}', '{data[1]}', '{data[2]}', '{data[3]}', '{data[4]}')"
    try:
        cursor.execute(insert_value_query)
    except Error as err:
        print(f"Error during insertion as :'{err}'")

def delete_values(cursor, delete_value_query):
    try:
        cursor.execute(delete_value_query)
    except Error as err:
        print(f"Error in deleting as : '{err}'")

def select_function(cursor, select_query):
    result = []
    try:
        cursor.execute(select_query)
        rows = cursor.fetchall()
        for row in rows:
            result.append(row)
    except Error as err:
        print(f"Error in selecting as : '{err}'")
    finally:
        return result

def select_values(cursor, select_query):
    """Insert values into the database
    :param cursor: the cursor object to execute queries
    :param insert_value_query the query to insert values into the table
    """
    try:
        cursor.execute(select_query)
    except Error as err:
        print(f"Error during selection as :'{err}'")


def main():
    
    print("currently in database:")
    select_statement = "SELECT * FROM predictions"
    print(select_function(cursor, select_statement))

    setup("91.121.93.94")
    while True:
        pass
        signal.signal(signal.SIGINT, handler)

if __name__ == '__main__':
   connection = create_connection(database)
   create_statement = "CREATE TABLE IF NOT EXISTS predictions(id INTEGER, dtimestp INTEGER, duration INTEGER, day VARCHAR, hour VARCHAR)"
   cursor = create_table(connection, create_statement)
   main()
   