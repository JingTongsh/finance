import pandas as pd
from main import load_data
import json
import numpy as np

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
    with open('metrics/ivs_0602.json', 'w') as f:
        json.dump(features_iv, f)
    print("IV values per feature:")
    print(features_iv)
    return features_iv


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data()
    # missing_rate(train_x)
    calculate_all_ivs(train_x, train_y)