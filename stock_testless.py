import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# CSVデータの読み込み
csv_list = ["TimeChart_cyber.csv", "TimeChart_JT.csv", "TimeChart_kose.csv"] # CSVファイルのリスト
data = pd.read_csv('csv/' + csv_list[1], encoding="UTF-8", index_col="日付", parse_dates=True)  # CSVファイルの読み込み

# データの整形
data = data.sort_values("日付") # 昔のデータから順番にsort
max_num = data["始値"].max() # "始値"のカラムの最大値
data["始値"] /= max_num # 0~1の範囲に変換
open_price = data["始値"]  # "日付"のカラムを取得

# 学習する過去何日分かのデータの作成をする関数の定義
interval = 20
def make_data(data, interval=20):
    input_tensor = [] # 学習データ
    output_tensor = [] # 結果
    open_price = list(data)
    for i in range(len(open_price)):
        if i < interval:
            continue
        output_tensor.append(open_price[i])
        z = []
        for j in range(interval):
            d = i + j -interval
            z.append(open_price[d])
        input_tensor.append(z)
    return (input_tensor, output_tensor)

# 関数の実行
X_train, y_train = make_data(open_price)

lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train) # 学習
predict_list = []
for i in range(4):
    X_test = []
    X_test.append(open_price[-1:-interval-1:-1])
    y_predict_small = lr.predict(X_test) # 予測
    predict_list.append(y_predict_small * max_num) # 0~1に変換した逆の変換
    open_price = list(open_price)
    open_price.append(y_predict_small)
    open_price = np.array(open_price)
    print("{}週間後: {}".format(i+1, predict_list[i][0]))