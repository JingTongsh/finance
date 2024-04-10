import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import joblib
import json


def load_data():
    # Load the training data
    train_x = pd.read_csv('data/train_x_202108.csv')
    train_y = pd.read_csv('data/train_y_202108.csv')

    # Load the testing data
    test_x = pd.read_csv('data/test_x_202109.csv')
    test_y = pd.read_csv('data/test_y_202109.csv')

    # remove object columns
    train_x = train_x.select_dtypes(exclude=['object'])
    train_y = train_y.select_dtypes(exclude=['object'])
    test_x = test_x.select_dtypes(exclude=['object'])
    test_y = test_y.select_dtypes(exclude=['object'])

    print('data loaded')

    return train_x, train_y, test_x, test_y


def evaluate_metrics(y_true, y_pred):
    # dict
    metrics = {'auc': roc_auc_score(y_true, y_pred)}
    thresh = 0.5
    y_pred_binary = y_pred > thresh
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['precision'] = precision_score(y_true, y_pred_binary)
    metrics['recall'] = recall_score(y_true, y_pred_binary)
    return metrics


def train_eval_xgboost():
    train_x, train_y, test_x, test_y = load_data()
    # Convert the data to DMatrix format
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)
    
    # Set the parameters for XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': 3,
        'eta': 0.1,
        'device': 'cuda',
    }

    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')])

    # Make predictions on the testing data
    preds = model.predict(dtest)

    # Evaluate the model
    metrics = evaluate_metrics(test_y, preds)
    with open('metrics_xgboost.json', 'w') as f:
        json.dump(metrics, f)
    print(f'metrics: {metrics}')

    # Save the trained model
    model.save_model('xgboost_model.bin')


def train_eval_sklearn(model_name: str):
    train_x, train_y, test_x, test_y = load_data()
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean') # You can choose other strategies such as 'median' or 'most_frequent'
    train_x = imputer.fit_transform(train_x)
    test_x = imputer.fit_transform(test_x)
    
    # Standardize the data
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)

    # convert target to 1d array
    train_y = train_y.values.ravel()
    test_y = test_y.values.ravel()
    
    models = {
        'logistic regression': LogisticRegression(),
        'random forest': RandomForestClassifier(),
        'gradient boosting': GradientBoostingClassifier(),
        'mlp': MLPClassifier(),
        'xgboost': XGBClassifier()
    }
    
    model = models[model_name]
    print(f'Training {model_name} ...')
    model.fit(train_x, train_y)

    # Make predictions on the testing data
    preds = model.predict_proba(test_x)[:, 1]

    # Evaluate the model
    metrics = evaluate_metrics(test_y, preds)
    with open(f'metrics_{model_name}.json', 'w') as f:
        json.dump(metrics, f)
    print(f'Done training {model_name}')
    print(f'metrics: {metrics}')

    # Save the trained model
    joblib.dump(model, f'{model_name}_model.pkl')


if __name__ == '__main__':
    # train_eval_xgboost()
    for model_name in ['logistic regression', 'random forest', 'gradient boosting', 'mlp', 'xgboost']:
        train_eval_sklearn(model_name)
        
    