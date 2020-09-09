import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Dropout

# CSVデータの読み込み
csv_list = ["TimeChart_cyber.csv", "TimeChart_JT.csv", "TimeChart_kose.csv"] # CSVファイルのリスト
data = pd.read_csv('csv/' + csv_list[2], encoding="UTF-8", index_col="日付", parse_dates=True)  # CSVファイルの読み込み

# データの整形
data = data.sort_values("日付") # 昔のデータから順番にsort
max_num = data["始値"].max()
data["始値"] /= max_num
X = data["始値"]

# 訓練データとテストデータに分割
train = X[:len(X) * 4 // 5]
test = X[len(X) * 4 // 5:]

# 学習する過去何日分かのデータの作成をする関数の定義
interval = 10
def make_data(data, interval=10):
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
X_train, y_train = make_data(train)
X_test, y_test = make_data(test)

# モデルの構造を定義
model = Sequential()
model.add(Dense(5, input_shape=(interval,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(1))
model.add(Activation('linear'))

model.summary()

# モデルを構築
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#モデルをfit
earlystopping = EarlyStopping(monitor='loss', patience=5)
model.fit(X_train, y_train, batch_size=10, epochs=50, callbacks=[earlystopping])

# 予測
predicted = model.predict(X_test)

# 最初の変換の逆変換
predicted_array = np.array(predicted)
predicted_array *= max_num

y_test_array = np.array(y_test)
y_test_array *= max_num

# グラフの設定と表示
result = pd.DataFrame(predicted_array)
result.columns = ['predict']
result['actual'] = y_test_array
result.plot()
plt.show()

# 正解率を表示
def updown(actual, predict):
    act = np.sign(np.diff(actual))
    pre = np.sign(np.diff(predict[:,0]))
    tmp =  act*pre>0
    return np.sum(tmp)/len(tmp)

print("正解率: ", updown(y_test, predicted) * 100, "%")

# １〜４週間後の株価を予測
predict_list = []
for i in range(4):
    X_pre = []
    X_pre.append(X[-1:-interval-1:-1])
    X_pre_array = np.asarray(X_pre)
    y_predict_small = model.predict(X_pre_array) # 予測
    result = y_predict_small * max_num
    predict_list.append(result)
    X = list(X)
    X.append(y_predict_small[0][0])
    X = pd.Series(X)
    print("{}週間後: {}".format(i+1, predict_list[i][0][0]))

latest_price = X[-5::].values[0] * max_num
rate_list = [""]
x = ["latest", "a week later", "2 weeks later", "3 weeks later", "4 weeks later"]
y = [latest_price]
for j in range(4):
    rate = (predict_list[j][0][0] - latest_price) / latest_price
    rate_round = round(rate * 100, 1)
    rate_list.append("(" + str(rate_round) + "%)")
    y.append(predict_list[j][0][0])
    
plt.figure(figsize=(10, 6), dpi=100)
plt.title("Stock Price Prediction Graph")
plt.ylabel("yen")
plt.plot(x, y, marker="D", markersize=5, markeredgewidth=3, markeredgecolor="blue", markerfacecolor="lightblue")
for (i, j, k) in zip(x, y, rate_list):
    detail_msg = str(int(round(j))) + "yen" + k
    plt.annotate(detail_msg, xy = (i, j+7.5), size = 10, color = "red")
plt.show()