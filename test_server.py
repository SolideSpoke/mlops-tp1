import requests

# host = "http://ubuntu@4.210.225.87:5001"
host = "http://localhost:8000"
r = requests.post(f"{host}/predict", json={"size": 100, "nb_rooms": 3, "garden": True})
print(r.json())