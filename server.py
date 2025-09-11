from fastapi import FastAPI
import joblib
from pydantic import BaseModel, HttpUrl
import tensorflow as tf
import requests
from io import BytesIO

app = FastAPI()
model_house = joblib.load("regression.joblib")
model_ship = tf.keras.models.load_model("ship.keras")
label_map = {
    'coastguard': 0, 'containership': 1, 'corvette': 2, 'cruiser': 3,
    'cv': 4, 'destroyer': 5, 'ferry': 6, 'methanier': 7,
    'sailing': 8, 'smallfish': 9, 'submarine': 10, 'tug': 11, 'vsmallfish': 12
}
class_names = [name for name, idx in sorted(label_map.items(), key=lambda item: item[1])]

class House(BaseModel):
    size: float
    nb_rooms: int
    garden: bool

class ShipRequest(BaseModel):
    img_url: HttpUrl

@app.get("/hello")
def hello():
    return {"message": "Hello World"}

@app.post("/predict")
def predict_house(house: House):
    prediction = model_house.predict([[house.size, house.nb_rooms, house.garden]])
    return {"y_pred": prediction[0]}

@app.get("/ship")
def predict_ship(ship: ShipRequest):
    response = requests.get(ship.img_url)
    img = BytesIO(response.content)
    img = tf.keras.utils.load_img(img, target_size=(32, 32))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model_ship.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return {
        "class": class_names[tf.argmax(score)],
        "confidence": float(100 * tf.reduce_max(score))
    }