from flask import Flask,jsonify,render_template,request
import config
from utils import Medical

app = Flask(__name__)

@app.route("/")
def get_homeapi():
    return "Welcome to VELOCITY"

@app.route("/Predict",methods=["POST","GET"])
def get_predict():
    if request.method == "POST":
        data = request.form 
        print(data)
        Pregnancies                 =  eval(data["Pregnancies"])
        Glucose                     =  eval(data["Glucose"])
        BloodPressure               =  eval(data["BloodPressure"])
        SkinThickness               =  eval(data["SkinThickness"])
        Insulin                     =  eval(data["Insulin"])
        BMI                         =  eval(data["BMI"])
        DiabetesPedigreeFunction    =  eval(data["DiabetesPedigreeFunction"])
        Age                         =  eval(data["Age"])
        
        medical2 = Medical(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
        classification = medical2.get_prediction()[0]
        return jsonify({"Result":f"Final Result is : {classification}"})


if __name__ =="__main__":
    app.run(host="0.0.0.0",port=config.PORT_NO)
