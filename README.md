# Disaster Response Pipeline Project
In this project, I analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
The training data for this project contains pre-labeled messages from various real life disaster events (related to weather, food, military, ...etc.). Some screenshots of the site and results are presented in the screenshots folder.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    For more information see the ETL_Pipeline_Preparation.ipynb notebook in the notebooks folder
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    For more information see the ML_Pipeline_Preparation.ipynb notebook in the notebooks folder

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Dependencies:
The project was build and testing using the following library versions:
- Python-3.6.3
- numpy-1.19.5 
- pandas-0.23.3
- scikit-learn-0.24.2
- Flask-0.12.5
- nltk-3.2.5
- plotly-2.0.15
- SQLAlchemy-1.2.19
- pickle-4.0

### Files Description:
1. app:
    - templates: contains html files to be used for rendering the app
    - run.py: script to run the Flask application
2. data:
    - DisasterResponse.db: the database that contains the cleaned data
    - disaster_categories.csv: csv file that contains the raw categories data
    - disaster_messages.csv: csv file that contains the raw message data
    - process_data.py: script to preprocess and clean the raw data and save it to the database
3. models:
    - train_classifier.py: script to load the data, and train different classifiers
4. notebooks:
    - ETL_Pipeline_Preparation.ipynb: Notebook that summerizes process_data.py
    - ML_Pipeline_Preparation.ipynb: Notebook that summerizes train_classifier.py
5. screenshots: 
    - contains screenshots of different examples of the application

### Acknowledgements:
The dataset was provided by Figure8, and the skelaton of the code was provided by Udacity 
