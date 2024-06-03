import pandas as pd
from main import load_data, new_load_data
import json
import numpy as np
from sklearn.impute import SimpleImputer
import xgboost as xgb

# compute the missing rate of each feature (training set)
def missing_rate(train_x):
    missing_rates = train_x.isnull().mean() * 100
    with open('metrics/missing_rates.json', 'w') as f:
        json.dump(missing_rates.to_json(), f)
    print("Missing rates per feature:")
    print(missing_rates)

def calculate_iv(df, feature, target_series, bins):
    data = pd.DataFrame({
        'Feature': df[feature],
        'Target': target_series.squeeze(),
    })
    # 等频分箱
    # data['bin'] = pd.qcut(data['Feature'], q=10, duplicates='drop')
    # 等宽分箱
    data['bin'] = pd.cut(data['Feature'], bins=bins)
    groups = data.groupby('bin')['Target'].agg(['count', 'sum'])
    groups['good'] = groups['count'] - groups['sum']
    groups['bad'] = groups['sum']
    groups['good_dist'] = groups['good'] / groups['good'].sum()
    groups['bad_dist'] = groups['bad'] / groups['bad'].sum()
    groups['iv'] = (groups['good_dist'] - groups['bad_dist']) * np.log((groups['good_dist'] + 0.0001) / (groups['bad_dist'] + 0.0001))
    iv = groups['iv'].sum()
    return iv

def calculate_all_ivs(df, target_series):
    # Initialize a dictionary to store IV values for all features
    features_iv = {}
    bins = 75      # best param
    # Loop over each column in the dataframe except the target series
    for col in df.columns:
        iv = calculate_iv(df, col, target_series, bins)
        features_iv[col] = iv
    with open('metrics/ivs_0602_08.json', 'w') as f:
        json.dump(features_iv, f)
    print("IV values per feature:")
    print(features_iv)
    return features_iv


def calculate_psi(expected, actual, bins=10):
    expected_percents, bins = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bins)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    psi = np.sum((expected_percents - actual_percents) * np.log((expected_percents + 0.0001) / (actual_percents + 0.0001)))
    return psi

def calculate_all_psi(df, target_series):
    # Initialize a dictionary to store IV values for all features
    features_iv = {}
    bins = 75      # best param
    # Loop over each column in the dataframe except the target series
    for col in df.columns:
        iv = calculate_psi(df[col], target_series[col], bins)
        features_iv[col] = iv
    with open('metrics/psi_0602_0809.json', 'w') as f:
        json.dump(features_iv, f)
    print("PSI values per feature:")
    print(features_iv)
    return features_iv


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data()
    # missing_rate(train_x)

    # ===== VI =====
    # 202108
    # calculate_all_ivs(train_x, train_y)
    # 202109
    # calculate_all_ivs(test_x, test_y)

    # ====== psi ======
    # # 缺失值处理，使用均值填充
    # num_imputer = SimpleImputer(strategy='mean')
    # train_x = pd.DataFrame(num_imputer.fit_transform(train_x), columns=train_x.columns)
    # test_x = pd.DataFrame(num_imputer.transform(test_x), columns=test_x.columns)
    # expected = train_x
    # actual = test_x
    # # 计算单个变量的PSI
    # psi_value = calculate_all_psi(expected, actual)

    # 加载模型
    model = xgb.XGBRegressor()
    model.load_model('/Users/zhitengli/proj/finance/models/xgboost_model_best.bin')
    # 预测概率
    train_x, train_y, test_x, test_y = new_load_data(activate_remove=0, f_num=165, b_num=75)
    # Convert the data to DMatrix format
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)
    y_train_pred = model.predict(train_x)
    y_test_pred = model.predict(test_x)
    # 计算模型预测值的PSI
    psi_model = calculate_psi(y_train_pred, y_test_pred)
    print(psi_model)