import requests

r = requests.post("http://localhost:8000/predict", json={"size": 100, "bedrooms": 3, "garden": True})
print(r.json())