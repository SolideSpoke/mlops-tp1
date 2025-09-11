import requests

# host = "http://ubuntu@4.210.225.87:8088"
host = "http://localhost:8002"
r = requests.post(f"{host}/predict", json={"size": 100, "nb_rooms": 3, "garden": True})
print(r.json())

ship = "https://www.dolphin-charger.fr/media/global/article-9.jpeg"
r = requests.get(f"{host}/ship", json={"img_url": ship})
print(r.json())