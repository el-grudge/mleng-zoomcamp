import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

def load_data():
    # load the data
    file_url = 'https://archive.ics.uci.edu/static/public/222/data.csv'
    df = pd.read_csv(file_url)
    return df

def prepare_data(df):
    ## data preparation
    # convert target var to numerical
    df.y = df.y.map({'yes':1,'no':0})

    # fill na
    df.fillna('unknown', inplace=True)

    # drop duration
    df.drop('duration', axis=1, inplace=True)

    # split the data into train/val/test with 80%/20%
    X_train, X_test = train_test_split(df, test_size=np.round(len(df)*.2).astype(int), random_state=42)
    
    y_train = X_train.y.values
    y_test = X_test.y.values

    del X_train['y']
    del X_test['y']

    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    # vectorize input
    X_train=X_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    dv.fit(X_train)
    
    X_train = dv.transform(X_train)

    ## model definition
    # Decision Tree
    model = DecisionTreeClassifier(max_depth=20, min_samples_leaf=1)
    model.fit(X_train, y_train)

    return dv, model

def evaluate(model, dv, X_test, y_test):
    X_test = X_test.to_dict(orient='records')
    X_test = dv.transform(X_test)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    print(f'precision={precision}')
    print(f'recall={recall}')
    print(f'f1={f1}')
    print(f'auc={auc}')

def save_model(dv, model):
    # save the model 
    max_depth = 20
    min_samples_leaf = 1
    output_file = f'model_depth={max_depth}_samples_leaf={min_samples_leaf}.bin'
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

    print(f'the model is saved to {output_file}')


if __name__ == "__main__":

    # load data
    df = load_data()

    # prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # train model 
    dv, model = train(X_train, y_train)
    
    # evaluate model 
    evaluate(model, dv, X_test, y_test)
    
    # save model 
    save_model(dv, model)
