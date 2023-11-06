import pandas as pd
import json
import pickle
from flask import Flask
from flask import request
from flask import jsonify
from utils import transform_data

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, data_transformer, model = pickle.load(f_in)

app = Flask('convert')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    customer = pd.DataFrame([customer])

    X = data_transformer(customer)
    X = dv.transform(X)
    y_pred = model.predict_proba(X)[0, 1]
    convert = y_pred >= 0.25

    result = {
        'conversion_probability': float(y_pred),
        'convert': bool(convert)
    }

    return jsonify(result)
    


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)