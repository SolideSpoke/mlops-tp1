from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("regression.joblib")

class House(BaseModel):
    size: float
    bedrooms: int
    garden: bool

@app.post("/predict")
def predict(house: House):
    prediction = model.predict([[house.size, house.bedrooms, house.garden]])
    return {"y_pred": prediction[0]}