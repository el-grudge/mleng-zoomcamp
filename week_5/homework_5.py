#%%
from subprocess import getoutput

# question 1
pipenv_version = getoutput("pipenv --version")
print(pipenv_version)

#%%
# question 2
import json

# Path to your Pipfile.lock
pipfile_lock_path = 'Pipfile.lock'

# Read the Pipfile.lock file
with open(pipfile_lock_path, 'r') as lock_file:
    lock_data = json.load(lock_file)

# Extract the Scikit-learn section
scikit_learn_section = lock_data.get('default', {}).get('scikit-learn')

# Print the Scikit-learn section
print(scikit_learn_section['hashes'][0])

#%%
# question 3
PREFIX="https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2023/05-deployment/homework"
os.system(f"wget {PREFIX}/model1.bin")
os.system(f"wget {PREFIX}/dv.bin")

import pickle


def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


dv = load('dv.bin')
model = load('model1.bin')

client = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform([client])
y_pred = model.predict_proba(X)[0, 1]

print(y_pred)