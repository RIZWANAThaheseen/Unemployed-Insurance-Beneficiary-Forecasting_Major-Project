from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
import plotly.express as px
#import os
#import os

#print("Templates folder exists:", os.path.exists(os.path.join(os.path.dirname(__file__), 'templates')))
#print("Index.html exists:", os.path.exists(os.path.join(os.path.dirname(__file__), 'templates', 'index.html')))


#app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))
app = Flask(__name__)


# Load the trained model
model_prophet = Prophet()
with open('Flask/model.pkl', 'rb') as file:
    model_prophet = pickle.load(file)

@app.route('/')
def home():
    # Render the index page
    return render_template('index.html')

@app.route('/inspect')
def inspect():
    # Show the inspect.html form for GET requests
    #if request.method == 'GET':
 return render_template('inspect.html')
    
    # Process the form data for POST requests
@app.route('/inspect', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        #try:
            # Get the user input date
            input_date = pd.to_datetime(request.form['input_date'])
            
            # Prepare the future date DataFrame for prediction
            future_date = pd.DataFrame({'ds': [input_date]})
            
            # Make the prediction
            forecast = model_prophet.predict(future_date)
            prediction = forecast['yhat'].values[0]

            # Example: Get the beneficiaries count (mock example)
            # Assuming 'data' is a pre-loaded DataFrame with beneficiaries info
            data = pd.DataFrame({'id': [1, 2, 3], 'beneficiary': [True, True, False]})
            beneficiaries_count = data['beneficiary'].sum()

            # Generate a line plot for the forecast
            fig = px.line(forecast, x='ds', y='yhat', title='Insurance Forecast')
            graph = fig.to_html(full_html=False)
            fig.write_html("test_graph.html")

            
            # Render output.html with the prediction, graph, and beneficiaries count
            return render_template('output.html', 
                                   prediction=round(prediction), 
                                   graph=graph, 
                                   beneficiaries_count=beneficiaries_count)
        
        #except Exception as e:
            # Render output.html with an error message if something goes wrong
            #return render_template('output.html', 
                                   #error=f"An error occurred: {str(e)}", 
                                  # beneficiaries_count=None, 
                                  # prediction=None, 
                                   #graph=None)
    return render_template('output.html', 
                           prediction=None, 
                           graph=None, 
                           beneficiaries_count=None)
    return render_template('index.html')

if __name__ == '__main__':
   app.run(debug=True)

