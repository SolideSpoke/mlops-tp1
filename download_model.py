import requests
from tqdm import tqdm

url = "https://epitafr-my.sharepoint.com/:u:/g/personal/aniss1_outaleb_epita_fr/ESo4I1UDjelEk38--3smuuwBQ8pMifV8q8VhKXalA9tqZg?e=lKVKZY&download=1"

response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open("ship.keras", "wb") as file, tqdm(
    desc="Downloading",
    total=total_size,
    unit='B',
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for data in response.iter_content(chunk_size=1024):
        file.write(data)
        bar.update(len(data))

print("File downloaded successfully.")