# Disaster Response Pipeline Project
In this project things take place:
- Short social media messages from disaster zones and their classifications are read into a dataframe.
- This data is transformed using nltk and scikit learn's pipeline.
- A model is built which predicts the classification of subsequent messages.

This can be viewed in a web app which runs locally. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# File Structure
- app
    - templates
        - go.html
        - master.html
    - run.py
- data
    - disaster_categories.csv
    - disaster_messages.csv
    - DisasterResponse.db
    - process_data.py
- models
    - classifier.pkl
    - train_classifier.py