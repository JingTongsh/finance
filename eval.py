import seaborn as sns
from matplotlib import ticker
from main import *

# load all models, plot roc in one figure

def load_models():
    models = {
        'gradient boosting': joblib.load('models/gradient boosting_model.pkl'),
        'logistic regression': joblib.load('models/logistic regression_model.pkl'),
        'random forest': joblib.load('models/random forest_model.pkl'),
        'mlp': joblib.load('models/mlp_model.pkl'),
        'xgboost': xgb.Booster()
    }
    models['xgboost'].load_model('models/xgboost_model.bin')
    return models


def main():
    models = load_models()
    train_x, train_y, test_x, test_y = load_data()
    
    # data for xgb
    dtest = xgb.DMatrix(test_x, label=test_y)
    
    # data for sklearn models
    imputer = SimpleImputer(strategy='mean')
    test_x = imputer.fit_transform(test_x)
    scaler = StandardScaler()
    test_x = scaler.fit_transform(test_x)
    test_y = test_y.values.ravel()
    
    # plot roc curve for all models
    fig, ax = plt.subplots()
    for model_name, model in models.items():
        preds = model.predict_proba(test_x)[:, 1] if model_name != 'xgboost' else model.predict(dtest)
        fpr, tpr, _ = roc_curve(test_y, preds)
        roc_auc = roc_auc_score(test_y, preds)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    plt.savefig('metrics/all_models.png')
    plt.show()
    plt.close()
    print('ROC curves saved to metrics/all_models.png')

def positive_proportion_curve(num_boxes: int = 100):
    """
    Load all model and predict on test data.
    in each box, show the proportion of positive samples and make a curve.
    """
    models = load_models()
    _, _, test_x, test_y = load_data()
    
    # data for xgb
    dtest = xgb.DMatrix(test_x, label=test_y)
    
    # data for sklearn models
    imputer = SimpleImputer(strategy='mean')
    test_x = imputer.fit_transform(test_x)
    scaler = StandardScaler()
    test_x = scaler.fit_transform(test_x)
    test_y = test_y.values.ravel()
    
    # plot roc curve for all models
    fig, ax = plt.subplots()
    for model_name, model in models.items():
        # 1是正样本
        preds = model.predict_proba(test_x)[:, 1] if model_name != 'xgboost' else model.predict(dtest)
        preds_sort = abs(np.sort(-preds))
        box_size = 1 / num_boxes
        positive_proportions = []
        # 前百分之x样本里正样本的比例
        for i in range(num_boxes):
            bound = (i+1) * box_size #前bound分数中正样本比例
            score = preds_sort[(int)(bound*len(preds_sort))-1]
            # print(score)
            y_p = 0
            y_a = 0
            for j in range(len(test_y)):
                if preds[j]>score:
                    y_a += 1
                    if test_y[j] == 1:
                        y_p += 1
            # print(y_a," ",y_p)
            positive_proportion = y_p/(y_a+0.0001)
            positive_proportions.append(positive_proportion)
        ax.plot(np.arange(num_boxes)/num_boxes, positive_proportions, label=f'{model_name}')
    
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_xlabel('Proportion of total samples')
    ax.set_ylabel('Proportion of positive samples')
    ax.set_title('Proportion of positive samples in the total sample')
    ax.legend(loc='best')
    
    plt.show()
    plt.savefig('metrics/positive_proportion_curve.png')
    plt.close()
    print('Proportion of positive samples in the total sample saved to metrics/positive_proportion_curve.png')

def box_proportion(num_boxes: int = 100):
    """
    Load xgboost model and predict on test data.
    Plot histogram of predicted probabilities;
    also in each box, show the proportion of positive samples.
    """
    model = xgb.Booster()
    model.load_model('models/xgboost_model.bin')
    train_x, train_y, test_x, test_y = load_data()
    dtest = xgb.DMatrix(test_x, label=test_y)
    preds = model.predict(dtest)
    
    # plot histogram of predicted probabilities
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
    axs[0].hist(preds, bins=num_boxes)
    axs[0].set_xlabel('Predicted probability')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Histogram of predicted probabilities')
    
    # plot proportion of positive samples in each box
    box_size = 1 / num_boxes
    box_proportions = []
    for i in range(num_boxes):
        lower_bound = i * box_size
        upper_bound = (i + 1) * box_size
        box_proportion = np.mean((preds >= lower_bound) & (preds < upper_bound))
        box_proportions.append(box_proportion)
    axs[1].bar(np.arange(num_boxes), box_proportions)
    axs[1].set_xlabel('Box number')
    axs[1].set_ylabel('Proportion of positive samples')
    axs[1].set_title('Proportion of positive samples in each box')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('metrics/box_proportion.png')
    plt.close()
    print('Histogram and box proportions saved to metrics/box_proportion.png')


def count_positive():
    """
    For train and test set, print number and proportion of positive samples.
    """
    train_x, train_y, test_x, test_y = load_data()
    train_y, test_y = train_y.values.ravel(), test_y.values.ravel()
    print(f'Train set: {train_y.sum()} positive samples out of {len(train_y)} ({train_y.mean() * 100:.2f}%)')
    print(f'Test set: {test_y.sum()} positive samples out of {len(test_y)} ({test_y.mean() * 100:.2f}%)')


def covariance_of_input_features():
    """
    Calculate the covariance matrix of the input features.
    """
    train_x, train_y, test_x, test_y = load_data()
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean') # You can choose other strategies such as 'median' or 'most_frequent'
    train_x = imputer.fit_transform(train_x)
    test_x = imputer.fit_transform(test_x)
    
    # Standardize the data
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    
    for s in ['train', 'test']:
        data = train_x if s == 'train' else test_x
        cov = np.cov(data, rowvar=False)
        print(f'Covariance matrix of {s} set:')
        # plot covariance matrix
        fig, ax = plt.subplots()
        sns.heatmap(cov, ax=ax, cmap='coolwarm')
        ax.set_title(f'Covariance matrix of {s} set')
        plt.tight_layout()
        plt.show()
        plt.savefig(f'metrics/covariance_{s}.png')
        plt.close()
        print(f'Covariance matrix saved to metrics/covariance_{s}.png')
        

def missing_rate():
    """
    Print missing rate for each feature.
    """
    train_x, train_y, test_x, test_y = load_data()
    for s in ['train', 'test']:
        data = train_x if s == 'train' else test_x
        missing_rate = data.isnull().mean()
        print(f'Missing rate of {s} set:')
        for col in data.columns:
            if missing_rate[col] > 0:
                print(f'{col}: {missing_rate[col] * 100:.2f}%')
        
        # plot missing rate
        fig, ax = plt.subplots()
        # ax.bar(data.columns, missing_rate)
        # don't show x-axis labels since there are too many features
        ax.bar(np.arange(len(data.columns)), missing_rate)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Missing rate')
        ax.set_title(f'Missing rate of {s} set')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'metrics/missing_rate_{s}.png')
        plt.close()


def correlation_with_output():
    """
    Calculate the correlation between each input feature and the output.
    """
    train_x, train_y, test_x, test_y = load_data()
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean') # You can choose other strategies such as 'median' or 'most_frequent'
    train_x = imputer.fit_transform(train_x)
    test_x = imputer.fit_transform(test_x)
    
    # Standardize the data
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    
    for s in ['train', 'test']:
        data = train_x if s == 'train' else test_x
        correlation = np.corrcoef(data, train_y if s == 'train' else test_y, rowvar=False)[-1, :-1]
        print(f'Correlation with output of {s} set:')
        
        # plot correlation
        fig, ax = plt.subplots()
        ax.bar(np.arange(data.shape[1]), correlation)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Correlation with output')
        ax.set_title(f'Correlation with output of {s} set')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'metrics/correlation_{s}.png')
        plt.close()
        print(f'Correlation with output saved to metrics/correlation_{s}.png')
        

if __name__ == '__main__':
    # correlation_with_output()
    # box_proportion(30)
    # count_positive()
    positive_proportion_curve(100)
    
    