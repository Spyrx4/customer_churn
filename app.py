# backend

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

# load artifacts
preprocess = joblib.load("artifacts/preprocessing_fe/preprocessing_artifacts.joblib")
model = joblib.load("artifacts/models/xgb_ros.pkl")

# tenure binning
TENURE_BINS = [0, 6, 12, 24, 60, np.inf]
TENURE_LABELS = list(range(len(TENURE_BINS) - 1))

def apply_binning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        right=False
    )
    return df


def inference_pipeline(raw_df: pd.DataFrame):
    df = apply_binning(raw_df)  
    
    # log transform
    for col in preprocess["log_cols"]:
        if col in df:
            df[col] = np.log1p(df[col])
    
    # scaling    
    df[preprocess["cont_cols"]] = preprocess["scaler"].transform(df[preprocess["cont_cols"]])
    
    # one hot encoding
    X = preprocess["ohe_preprocess"].transform(df)
    
    # predict
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    
    return int(pred[0]), float(proba[0])

# fastapi
app = FastAPI(title="Customer churn model API") 

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid input format",
            "details": exc.errors(),
            "received_body": exc.body,
        }
    )
    
# input schema
class CustInput(BaseModel):
    gender: str = Field(alias="gender")
    SeniorCitizen: int
    Partner: str = Field(alias="Partner")
    Dependents: str = Field(alias="Dependents")
    tenure: int
    PhoneService: str = Field(alias="PhoneService")
    MultipleLines: str = Field(alias="MultipleLines")
    InternetService: str = Field(alias="InternetService")
    OnlineSecurity: str = Field(alias="OnlineSecurity")
    OnlineBackup: str = Field(alias="OnlineBackup")
    DeviceProtection: str = Field(alias="DeviceProtection")
    TechSupport: str = Field(alias="TechSupport")
    StreamingTV: str = Field(alias="StreamingTV")
    StreamingMovies: str = Field(alias="StreamingMovies")
    Contract: str = Field(alias="Contract")
    PaperlessBilling: str = Field(alias="PaperlessBilling")
    PaymentMethod: str = Field(alias="PaymentMethod")
    MonthlyCharges: float
    TotalCharges: float
    
    class Config:
        populate_by_name = True
        
# prediction endpoint
@app.post("/predict")
def predict(data: CustInput):
    
    raw = pd.DataFrame([{
        'gender': data.gender,
        'SeniorCitizen': data.SeniorCitizen,
        'Partner': data.Partner,
        'Dependents': data.Dependents,
        'tenure': data.tenure,
        'PhoneService': data.PhoneService,
        'MultipleLines': data.MultipleLines,
        'InternetService': data.InternetService,
        'OnlineSecurity': data.OnlineSecurity,
        'OnlineBackup': data.OnlineBackup,
        'DeviceProtection': data.DeviceProtection,
        'TechSupport': data.TechSupport,
        'StreamingTV': data.StreamingTV,
        'StreamingMovies': data.StreamingMovies,
        'Contract': data.Contract,
        'PaperlessBilling': data.PaperlessBilling,
        'PaymentMethod': data.PaymentMethod,
        'MonthlyCharges': data.MonthlyCharges,
        'TotalCharges': data.TotalCharges
    }])
    
    pred, proba = inference_pipeline(raw)
    
    risk = (
        "High" if proba >= 0.8 else
        "Medium" if proba >= 0.5 else
        "Low"
    )
    
    return {
        "prediction": pred,
        "churn_probability": proba,
        "risk_level": risk
    }
    
# health check
@app.get("/")
def health():
    return {"status": "API is running"}