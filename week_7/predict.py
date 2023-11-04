import pickle
from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_depth=20_samples_leaf=1.bin'

print('model_file')

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('convert')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    convert = y_pred >= 0.5

    result = {
        'conversion_probability': float(y_pred),
        'convert': bool(convert)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)