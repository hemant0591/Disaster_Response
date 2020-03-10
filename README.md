# Introduction
This project aims to build a text processing pipeline so that the overwhelming number of messages received during the disaster time can be classified easily and effectively into the proper categories.


# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
     
     After running this command in the project root directory a sql database will be created which will store the data to be used later in
     building the ML pipeline.
     
     
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
     This commands trains the model, evaluates it and saves it in a pickle file to be used later for categorization of messages.

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
   This file will create a link (http://localhost:2020) to the local host which will lead to a web page which hosts the dash boards as      well as a search bar for 
   categorization of key words or phrases

