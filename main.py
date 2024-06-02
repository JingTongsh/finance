import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
import joblib
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import KBinsDiscretizer


def load_data():
    # 加载训练数据和测试数据
    train_x = pd.read_csv('data/train_x_202108.csv')
    train_y = pd.read_csv('data/train_y_202108.csv')
    test_x = pd.read_csv('data/test_x_202109.csv')
    test_y = pd.read_csv('data/test_y_202109.csv')

    # 删除所有值都为缺失的列
    train_x = train_x.dropna(axis=1, how='all')
    test_x = test_x.dropna(axis=1, how='all')

    # 仅保留数值型列
    train_x = train_x.select_dtypes(exclude=['object'])
    train_y = train_y.select_dtypes(exclude=['object'])
    test_x = test_x.select_dtypes(exclude=['object'])
    test_y = test_y.select_dtypes(exclude=['object'])

    return train_x, train_y, test_x, test_y

def new_load_data(activate_remove=True, f_num=10, b_num=2):
    # 加载训练数据和测试数据
    train_x = pd.read_csv('data/train_x_202108.csv')
    train_y = pd.read_csv('data/train_y_202108.csv')
    test_x = pd.read_csv('data/test_x_202109.csv')
    test_y = pd.read_csv('data/test_y_202109.csv')

    # 删除所有值都为缺失的列
    train_x = train_x.dropna(axis=1, how='all')
    test_x = test_x.dropna(axis=1, how='all')

    # 仅保留数值型列
    train_x = train_x.select_dtypes(exclude=['object'])
    train_y = train_y.select_dtypes(exclude=['object'])
    test_x = test_x.select_dtypes(exclude=['object'])
    test_y = test_y.select_dtypes(exclude=['object'])

    # 缺失值处理，使用均值填充
    num_imputer = SimpleImputer(strategy='mean')
    train_x = pd.DataFrame(num_imputer.fit_transform(train_x), columns=train_x.columns)
    test_x = pd.DataFrame(num_imputer.transform(test_x), columns=test_x.columns)

    # 特征选择 - 移除低方差特征
    if activate_remove:
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        train_x = pd.DataFrame(sel.fit_transform(train_x), columns=train_x.columns[sel.get_support()])
        test_x = pd.DataFrame(sel.transform(test_x), columns=test_x.columns[sel.get_support()])

    # 选择与目标变量相关性最高的100个特征
    selector = SelectKBest(f_classif, k=f_num)
    train_x = pd.DataFrame(selector.fit_transform(train_x, train_y), columns=train_x.columns[selector.get_support()])
    test_x = pd.DataFrame(selector.transform(test_x), columns=test_x.columns[selector.get_support()])

    # 分箱处理，将数值特征进行等宽分箱
    est = KBinsDiscretizer(n_bins=b_num, encode='ordinal', strategy='uniform') # 等距
    # est = KBinsDiscretizer(n_bins=40, encode='ordinal', strategy='quantile') # 等频分箱
    train_x = est.fit_transform(train_x)
    test_x = est.transform(test_x)

    return train_x, train_y, test_x, test_y


def evaluate_metrics(y_true, y_pred):
    # dict
    all_metrics = {}
    for thresh in np.arange(0.1, 1.0, 0.1):
        metrics = {'auc': roc_auc_score(y_true, y_pred)}
        # 可以选择用这个thrshold
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ks = max(abs(fpr - tpr))
        metrics["KS"] = ks.astype('float')
        y_pred_binary = y_pred > thresh
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['precision'] = precision_score(y_true, y_pred_binary)
        metrics['recall'] = recall_score(y_true, y_pred_binary)
        all_metrics[f'threshold {thresh}'] = metrics
    return all_metrics


def plot_roc_curve(y_true, y_pred, title: str, save_file: str):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.savefig(save_file)
    plt.show()
    plt.close()
    print(f'ROC curve saved to {save_file}')


def plot_ks_curve(y_true, y_pred, title: str, save_file: str):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = max(abs(fpr - tpr))
    plt.plot(fpr, label='bad')
    plt.plot(tpr, label='good')
    plt.plot(abs(fpr - tpr), label='diff')
    # 标记ks
    x = np.argwhere(abs(fpr - tpr) == ks)[0, 0]
    plt.plot((x, x), (0, ks), label='ks - {:.2f}'.format(ks), color='r', marker='o', markerfacecolor='r', markersize=5)
    plt.scatter((x, x), (0, ks), color='r')
    plt.legend()
    plt.savefig(save_file)
    plt.show()
    plt.close()
    print(f'KS curve saved to {save_file}')


def train_eval_xgboost():
    train_x, train_y, test_x, test_y = load_data()
    # Convert the data to DMatrix format
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    # Set the parameters for XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': 6,
        'eta': 0.1,
        'device': 'cuda',
        'lambda': 0.5,
    }

    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')])

    # Make predictions on the testing data
    preds = model.predict(dtest)

    # Evaluate the model
    title = 'XGBoost ROC curve'
    roc_curve_file = 'metrics/xgboost-tuned.png'
    # plot_roc_curve(test_y, preds, title, roc_curve_file)
    title = 'XGBoost KS curve'
    ks_curve_file = 'metrics/xgboost-ks.png'
    # plot_ks_curve(test_y, preds, title, ks_curve_file)
    metrics = evaluate_metrics(test_y, preds)
    with open('metrics/xgboost-tuned.json', 'w') as f:
        json.dump(metrics, f)
        print(f'Metrics saved to metrics/xgboost-tuned.json')
    print(f'metrics: {metrics}')

    # Save the trained model
    model.save_model('models/xgboost_model.bin')


def train_eval_xgboost_grid_search():
    max_ks = 0.0
    max_auc = 0.0

    best_activate = -1
    best_feature_num = 0
    best_bin_num = 0

    for i in range(2):
        for feature_select_num in range(50, 201, 5):
            for bin_num in range(5, 101, 5):
                train_x, train_y, test_x, test_y = new_load_data(activate_remove=i, f_num=feature_select_num, b_num=bin_num)
                # Convert the data to DMatrix format
                dtrain = xgb.DMatrix(train_x, label=train_y)
                dtest = xgb.DMatrix(test_x, label=test_y)

                # Set the parameters for XGBoost
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': ['logloss', 'auc'],
                    'max_depth': 6,
                    'eta': 0.1,
                    'device': 'cuda',
                    'lambda': 0.5,
                }

                model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')])

                # Make predictions on the testing data
                preds = model.predict(dtest)

                # Evaluate the model
                title = 'XGBoost ROC curve'
                roc_curve_file = 'metrics/xgboost-tuned.png'
                # plot_roc_curve(test_y, preds, title, roc_curve_file)
                title = 'XGBoost KS curve'
                ks_curve_file = 'metrics/xgboost-ks.png'
                # plot_ks_curve(test_y, preds, title, ks_curve_file)
                metrics = evaluate_metrics(test_y, preds)
                with open('metrics/xgboost-tuned.json', 'w') as f:
                    json.dump(metrics, f)
                    # print(f'Metrics saved to metrics/xgboost-tuned.json')
                print(f'metrics: {metrics}')
                if metrics["threshold 0.1"]["auc"] > max_auc and metrics["threshold 0.1"]["KS"] > max_ks:
                    max_auc = metrics["threshold 0.1"]["auc"]
                    max_ks = metrics["threshold 0.1"]["KS"]
                    best_activate = i
                    best_feature_num = feature_select_num
                    best_bin_num = bin_num
                print(f"Current Option: activate:{i}, select_num:{feature_select_num}, bin_num:{bin_num}")
                print(f"Best option: activate:{best_activate}, select_num:{best_feature_num}, bin_num:{best_bin_num}")
                print(f"max_auc={max_auc} max_ks={max_ks}")
                # Save the trained model
                model.save_model('models/xgboost_model.bin')


def train_eval_sklearn(model_name: str):
    train_x, train_y, test_x, test_y = load_data()

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')  # You can choose other strategies such as 'median' or 'most_frequent'
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
        'logistic regression': LogisticRegression(C=1e-3, penalty='l2'),
        'random forest': RandomForestClassifier(),
        'gradient boosting': GradientBoostingClassifier(),
        'mlp': MLPClassifier(hidden_layer_sizes=[2048]),
        'xgboost': XGBClassifier()
    }

    model = models[model_name]
    print(f'Training {model_name} ...')
    model.fit(train_x, train_y)

    # Make predictions on the testing data
    preds = model.predict_proba(test_x)[:, 1]

    # Evaluate the model
    file_name = model_name.replace(' ', '_')
    title = f'{model_name} ROC curve'
    roc_curve_file = f'metrics/{file_name}.png'
    plot_roc_curve(test_y, preds, title, roc_curve_file)
    title = 'XGBoost KS curve'
    ks_curve_file = f'metrics/{file_name}-ks.png'
    plot_ks_curve(test_y, preds, title, ks_curve_file)
    metrics = evaluate_metrics(test_y, preds)
    with open(f'metrics/{file_name}.json', 'w') as f:
        json.dump(metrics, f)
        print(f'Metrics saved to metrics/{file_name}.json')
    print(f'Done training {model_name}')
    print(f'metrics: {metrics}')

    # Save the trained model
    joblib.dump(model, f'models/{model_name}_model.pkl')


if __name__ == '__main__':
    for dir_name in ['models', 'metrics']:
        os.makedirs(dir_name, exist_ok=True)
    # train_eval_xgboost()
    train_eval_xgboost_grid_search()
    # for model_name in ['svc', 'logistic regression', 'random forest', 'gradient boosting', 'mlp', 'xgboost']:
    #     train_eval_sklearn(model_name)
