import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb


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


def transform_data(df=pd.DataFrame()):
    num = ['age', 'balance', 'day_of_week', 'campaign', 'pdays', 'previous']
    cat = ['job', 'housing', 'contact', 'month', 'poutcome']
    features = num + cat
    df_transformed =  df[features].copy()

    # mapping pdays based on conditions
    df_transformed['pdays'] = df_transformed['pdays'].apply(lambda x: 'never' if x == -1 else ('plus 12 months' if x > 365 else ('plus 6 months' if 180 <= x <= 365 else '6 months')))
    # mapping previous based on conditions
    df_transformed['previous'] = df_transformed['previous'].apply(lambda x: 'never' if x == 0 else ('more than 5' if x > 5 else 'less than 5'))
    # mapping campaing based on conditions
    df_transformed['campaign'] = df_transformed['campaign'].apply(lambda x: 'once' if x == 1 else 'more than once')

    # Consolidate categories of categorical features with many categories
    seasons = {
        'fall': ['sep','oct','nov'],
        'winter': ['dec','jan','feb'],
        'spring': ['mar','apr','may'],
        'summer': ['jun','jul','aug']
    }

    df_transformed['season'] = [season[0] for mon in df_transformed['month'] for season in list(seasons.items()) if mon in season[1]]
    
    job_category = {
        'cat_1': ['blue-collar','entrepreneur','housemaid'],
        'cat_2': ['retired','student','unemployed'],
        'cat_3': ['technician', 'admin.', 'management', 'services','unknown', 'self-employed']
    }

    df_transformed['job_category'] = [category[0] for job in df_transformed['job'] for category in list(job_category.items()) if job in category[1]]
    df_transformed = df_transformed.drop(['month','job'], axis=1).copy()
    df_transformed['contact'] = ['no' if contact == 'unknown' else 'yes' for contact in df_transformed['contact']]

    df_transformed['poutcome'] = [outcome if outcome in ['success', 'failure'] else 'other' for outcome in df_transformed['poutcome']]

    return df_transformed.to_dict(orient='records')


def train(X_train, y_train):
    # vectorize input
    dv = DictVectorizer(sparse=False)
    dv.fit(X_train)
    
    X_train = dv.transform(X_train)
    
    ## model definition
    # Decision Tree
    model = xgb.XGBClassifier(eta=0.3, max_depth=3, min_child_weight=1, objective='binary:logistic', eval_metric='auc', random_state=42)        
    model.fit(X_train, y_train)

    return dv, model


def evaluate(model, dv, X_test, y_test):
    X_test = dv.transform(X_test)
    y_pred = model.predict_proba(X_test)[:,1]
    precision = precision_score(y_test, y_pred >= 0.25, zero_division=0)
    recall = recall_score(y_test, y_pred >= 0.25)
    f1 = f1_score(y_test, y_pred >= 0.25, zero_division=0)
    auc = roc_auc_score(y_test, y_pred >= 0.25)
    
    print(f'precision={precision}')
    print(f'recall={recall}')
    print(f'f1={f1}')
    print(f'auc={auc}')


def save_model(dv, transform_data, model):
    # save the model 
    output_file = 'model.bin'
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, transform_data, model), f_out)

    print(f'the model is saved to {output_file}')


