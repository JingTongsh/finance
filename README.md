# 金融服务计算E1课题：违约预测

2024-04

## 数据准备

把数据放在 `data` 文件夹下，或使用软链接
```shell
ln -s /path/to/data data
```

此文件夹内含有所需的 `csv` 文件。

## python 环境

此环境经过测试能用
```shell
# （可选）新建并激活
conda create -n xgb python=3.11.5
conda activate xgb
# 安装所需库
pip install -U pandas scikit-learn xgboost
# 或
pip install -r requirements.txt
```

## 训练及测试

直接运行代码即可
```shell
python main.py
```

代码会自动在 `metrics` 文件夹记录评价指标，在 `models` 文件夹存储训练好的模型。

## Result

| Method | AUC |
| ------ | --- |
| logistic regression | 0.698 |
| mlp | 0.637 |
| random forest | 0.594 |
| xgboost | **0.706** |
| gradient boosting | 0.623 |

## TODO

* recall很低，可能需要调阈值
* 在测试数据 `data/data_x_202110.csv` 预测结果，上传打分
* 计算其他各种针对变量或针对模型的评价指标
* 按要求画图
* ……
