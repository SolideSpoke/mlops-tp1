from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("regression.joblib")

class House(BaseModel):
    size: float
    nb_rooms: int
    garden: bool

@app.get("/hello")
def hello():
    return {"message": "Hello World"}

@app.post("/predict")
def predict(house: House):
    prediction = model.predict([[house.size, house.nb_rooms, house.garden]])
    return {"y_pred": prediction[0]}