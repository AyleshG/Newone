import config
import pickle
import json
import numpy as np

class Medical():

    def __init__(self,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
        self.Pregnancies                = Pregnancies
        self.Glucose                    = Glucose
        self.BloodPressure              = BloodPressure
        self.SkinThickness              = SkinThickness
        self.Insulin                    = Insulin
        self.BMI                        = BMI
        self.DiabetesPedigreeFunction   = DiabetesPedigreeFunction
        self.Age                        = Age

    def get_load_model(self):
        with open(config.KNN_MODEL_PATH,"rb") as f:
            self.model=pickle.load(f)
        with open(config.STD_SCALER_MODEL1_PATH,"rb") as f:
            self.std=pickle.load(f)
        with open (config.LABELLED_DATA1_PATH,"r") as f:
            self.json_data=json.load(f)

    def get_prediction(self):
        self.get_load_model()
        test_array = np.zeros(len(self.json_data["columns"]))

        test_array[0] = self.Pregnancies
        test_array[1] = self.Glucose
        test_array[2] = self.BloodPressure
        test_array[3] = self.SkinThickness
        test_array[4] = self.Insulin
        test_array[5] = self.BMI
        test_array[6] = self.DiabetesPedigreeFunction
        test_array[7] = self.Age

        std_array1 = self.std.transform([test_array])

        predict = self.model.predict(std_array1)
        #print("Prediction Value is : ",predict)
        return predict

if __name__=="__main__":
        
        Pregnancies                 =  6.000
        Glucose                     =  148.000
        BloodPressure               =  72.000
        SkinThickness               =  35.000
        Insulin                     =  0.000
        BMI                         =  33.600
        DiabetesPedigreeFunction    =  0.627
        Age                         =  50.000


        medical_1 = Medical(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
        medical_1.get_prediction()


