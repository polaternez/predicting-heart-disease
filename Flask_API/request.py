import requests

url = "http://localhost:5000/predict_api"
data = {"input": [49, 'F', 'NAP', 160, 180, 0, 'Normal', 156, 'N', 1.0, 'Flat']}
r = requests.post(url, json=data)

print(r.text)
