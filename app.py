from flask import Flask, render_template, request
from model import get_response,predict_class
import json
import joblib
import pandas as pd
app=Flask(__name__)

#chatbot intents
intents = json.loads(open("intents.json").read())

#chatbot response route


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get')
def bot_response():
    userTxt=request.args.get('msg')
    
    ints=predict_class(userTxt)
    return get_response(ints,intents)


@app.route('/mo1ab')
def mo1ab():
    return render_template('mo1ab.html')

@app.route('/no2ba')
def no2ba():
    return render_template('no2ba.html')

def prediction_m01ab(x_data):
    #prediction
    #get pickled model
    demand_predictor=joblib.load('./models/reg_model.pkl')
    #x_data=[2019,11,4]
    input_df=pd.DataFrame([x_data],columns=['year','month','day'])
    #input_df
    res=round(demand_predictor.predict(input_df)[0],2)
    return res

def prediction_n02ba(x_data):
    #prediction
    #get pickled model
    demand_predictor=joblib.load('./models/reg_model_n02ba.pkl')
    #x_data=[2019,11,4]
    input_df=pd.DataFrame([x_data],columns=['year','month','day'])
    #input_df
    res=round(demand_predictor.predict(input_df)[0],2)
    return res

@app.route('/demand-m01ab', methods=['POST'])
def result():
    if request.method == 'POST':
        inputs=request.form.to_dict()
        inputs=list(inputs.values())
        inputs=list(map(int,inputs))
        y_hat=prediction_m01ab(inputs)
        
        return render_template('demandm01ab.html',y_hat=y_hat)
    
@app.route('/demand-n02ba', methods=['POST'])
def result2():
    if request.method == 'POST':
        inputs=request.form.to_dict()
        inputs=list(inputs.values())
        inputs=list(map(int,inputs))
        y_hat=prediction_n02ba(inputs)
        
        return render_template('demandn02ba.html',y_hat=y_hat)
    

    

if __name__=='__main__':
    app.run(debug=True)
    