import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path =model_path)
            preprocessor = load_object(file_path =preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        age:int,
        workclass:str,
        fnlwgt:int,
        education:str,
        educationnum:int,
        maritalstatus:str,
        Occupation:str,
        relationship:str,
        race:str,
        sex:str,
        capitalgain:int,
        capitalloss:int,
        hoursperweek:int,
        country:str
    ):
        self.age = age
        self.workclass = workclass
        self.fnlwgt = fnlwgt
        self.education = education
        self.educationnum = educationnum
        self.maritalstatus = maritalstatus
        self.Occupation = Occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capitalgain = capitalgain
        self.capitalloss = capitalloss
        self.hoursperweek = hoursperweek
        self.country = country

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age":[self.age],
                "workclass":[self.workclass],
                "fnlwgt":[self.fnlwgt],
                "education":[self.education],
                "educationnum":[self.educationnum],
                "maritalstatus":[self.maritalstatus],
                "Occupation":[self.Occupation],
                "relationship":[self.relationship],
                "race":[self.race],
                "sex":[self.sex],
                "capitalgain":[self.capitalgain],
                "capitalloss":[self.capitalloss],
                "hoursperweek":[self.hoursperweek],
                "country":[self.country]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)

