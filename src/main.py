from fastapi import FastAPI
import pickle, uvicorn, os
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import accuracy_score

# Config & Setup
## Variables of environment
DIRPATH = os.path.dirname(__file__)
ASSETSDIRPATH = os.path.join(DIRPATH, "asset")
ml_comp_pkl = os.path.join(ASSETSDIRPATH, "ml_comp.pkl")

print(
    f" {'*'*10} Config {'*'*10}\n INFO: DIRPATH = {DIRPATH} \n INFO: ASSETSDIRPATH = {ASSETSDIRPATH} "
)


# API Basic config
app = FastAPI(
    title="Titanic Survivors API",
    version="0.0.1",
    description="Prediction of Titanic Survivors",
)

## Loading of assets
with open(ml_comp_pkl, "rb") as f:
    loaded_items = pickle.load(f)
#print("INFO:    Loaded assets:", loaded_items)

pipeline_of_my_model = loaded_items["pipeline"]
num_cols = loaded_items['numeric_columns']
cat_cols = loaded_items['categorical_columns']

## BaseModel
class ModelInput(BaseModel):
    PeopleInTicket: int
    Age: float
    FarePerPerson: float
    SibSp: int
    Pclass: int
    Fare: float
    Parch: int
    TicketNumber: float
    Embarked: str
    Sex: str
    Title: str

## Utils
# def processing_FE(
#     dataset, scaler, encoder,imputer, FE=pipeline_of_my_model
# ):  # FE : ColumnTransfromer, Pipeline
#     "Cleaning, Processing and Feature Engineering of the input dataset."
#     """:dataset pandas.DataFrame"""

#     # if imputer is not None:
#     #     output_dataset = imputer.transform(dataset)
#     # else:
#     #     output_dataset = dataset.copy()

#     # output_dataset = scaler.transform(output_dataset)

#     # if encoder is not None:
#     #     output_dataset = encoder.transform(output_dataset)
#     if FE is not None:
#         output_dataset = FE.fit(output_dataset)

#     return output_dataset



def make_prediction(
     Pclass, Sex, Age, SibSp,Parch, Fare, Embarked, PeopleInTicket, FarePerPerson, TicketNumber, Title
    
):
    "Function to make one prediction"

    data = {
    "PeopleInTicket": PeopleInTicket,
    "Age": Age,
    "FarePerPerson": FarePerPerson,
    "SibSp": SibSp,
    "Pclass": Pclass,
    "Fare": Fare,
    "Parch": Parch,
    "TicketNumber": TicketNumber,
    "Embarked": Embarked ,
    "Title": Title ,
    "Sex": Sex }

    df = pd.DataFrame([data])
        
    X = df
    output = pipeline_of_my_model.predict(X).tolist()
    return output
   
   


## Endpoints
@app.post("/Titanic")
async def predict(input: ModelInput):
    """__descr__
    --details---
    """
    output_pred = make_prediction(
        PeopleInTicket =input.PeopleInTicket,
        Age =input.Age,
        FarePerPerson =input.FarePerPerson,
        SibSp =input.SibSp,
        Pclass =input.Pclass,
        Fare =input.Fare,
        Parch =input.Parch,
        TicketNumber =input.TicketNumber,
        Embarked =input.Embarked,
        Sex=input.Sex,
        Title=input.Title,
    )
     # Labelling Model output
    if output_pred == 0:
        output_pred = "No,the person didn't survive"
    else:
        output_pred = "Yes,the person survived"
    #return output_pred
    return {
        "prediction": output_pred,
        "input": input
    }

@app.post("/eaedk")
async def predict(input: ModelInput):
    with open('src/asset/ml_comp.pkl', 'rb') as file:
        loaded_object = pickle.load(file)

    data = {
    "PeopleInTicket": 0,
    "Age": 0,
    "FarePerPerson": 0,
    "SibSp": 0,
    "Pclass": 0,
    "Fare": 0,
    "Parch": 0,
    "TicketNumber": 0,
    "Embarked": "S",
    "Title": "Mr",
    "Sex": "male"}

    df = pd.DataFrame([data])

    pred = loaded_object['pipeline'].predict(df).tolist()

    # Labelling Model output
    if pred == 0:
        pred = "No,the person didn't survive"
    else:
        pred = "Yes,the person survived"
    #return pred
    return {
        "prediction": pred,
        "input": df.to_dict()#['records']
    }

    return pred

# Execution

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        reload=True,
    )