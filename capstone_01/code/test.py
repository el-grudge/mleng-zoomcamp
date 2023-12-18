import requests

url = 'http://localhost:9696/predict'

data = {'url': 'https://raw.githubusercontent.com/el-grudge/mleng-zoomcamp/main/capstone_01/cairo_frame1050.jpg'}

result = requests.post(url, json=data).json()
print(result)
