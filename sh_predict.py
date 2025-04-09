import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard,LearningRateScheduler, Callback, EarlyStopping


# 1 LearningRateLoggerの定義（カスタムコールバック）
class LearningRateLogger(Callback):
    def __init__(self):
        super(LearningRateLogger, self).__init__()
        self.lr_logs = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr.numpy()
        self.lr_logs.append(lr)
        print(f"\nEpoch {epoch+1}: Learning rate = {lr:.6f}")



# 2 Transformer用のEncoderレイヤーを定義
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"),
             tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 3 データの読み込み
df = pd.read_csv("C:/Users/sunak/OneDrive/ドキュメント/心不全予測データセット.csv")

# 4 列を数値型に変換  必要に応じて
#例df['Close'] = pd.to_numeric(df['Close'])

# 自前クリエイトデータセット関数、ライブラリで使えるやつあるかも？まあわからんので定義しよう！
def create_dataset(df, features, target_column='Target'):
    """
    DataFrameから特徴量とターゲット変数を作成する関数（正規化、分割済みデータ対応）

    Args:
        df (pd.DataFrame): データセットのDataFrame
        features (list): 特徴量の列名リスト
        target_column (str): ターゲット変数の列名 (デフォルト: 'Target')

    Returns:
        tuple: 特徴量のNumPy配列 (dataX) とターゲット変数のNumPy配列 (dataY)
    """

    dataX = df[features].values
    dataY = df[target_column].values
    return dataX, dataY

# 5 ターゲット変数の設定
df['Target'] = df['HeartDisease']
df.dropna(inplace=True)

"""
 6特徴量エンジニアリング 必要に応じて 男性女性などのカテゴリカル変数は順序関係がないならonehotエンコーディングで
 順序関係があるなら一意の数を割り振り!0,1とか,
 今回の場合運動誘発性狭心症の罹患歴に関しては順序関係があるかもしれないので予測に影響を与える可能性があるので、ラベルエンコーディングで試している。
 onehotエンコーディングと両方試すほうがいいかも。
"""

# 7 使用する特徴量 famale とmaleは明らかに負の相関があるため片方のみ採用　詳しくは　心不全多変量解析より
features = ['Age', 'Male','RestingBP','Cholesterol','FastingBS','MaxHR',"ExerciseAngina","Oldpeak"] 


# 8 データをnumpy配列に変換 ここで特徴量とか以外のいらない列は消える！
data = df[features].values

# 9 正規化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 10 訓練データとテストデータに分割
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
   #0からtrainsizeまでとtrainsizeから残りまで

# 11データセットの作成
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]
scaled_df = pd.DataFrame(scaled_data, columns=features)
scaled_df['Target'] = df['Target'].values # ターゲット変数を結合（スケーリングは不要）

train_df_scaled = scaled_df.iloc[0:train_size]
test_df_scaled = scaled_df.iloc[train_size:len(scaled_data)]

trainX, trainY = create_dataset(train_df_scaled, features)
testX, testY = create_dataset(test_df_scaled, features)



#確認用いらんかったらコメントアウト推奨
print("trainX shape:", trainX.shape)
print("trainY shape:", trainY.shape)
print("testX shape:", testX.shape)
print("testY shape:", testY.shape)



# 12 過去のデータ数  過去データを参照する場合　時系列等でなければ必要なしかな
#　13なし
# 14 データの形状をTransformerに合うように変更
# (サンプル数, 過去データ数, 特徴量数)!!!!!　時系列でない場合は過去データ数のところを１に！２重かっこ内が左のかっこの内容と合致している
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print("Reshaped trainX shape:", trainX.shape)
print("Reshaped testX shape:", testX.shape)
# 15 Transformerモデルの構築　
input_shape = (trainX.shape[1], len(features))
inputs = Input(shape=input_shape)
x = inputs

# 16 Transformerのパラメータ
embed_dim = len(features) 
num_heads = 2  
ff_dim = 32  

# 17 Transformer Encoder層を複数積み重ねる
for _ in range(3):
    x = TransformerEncoder(embed_dim, num_heads, ff_dim)(x)

x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x) # 正則化
x = Dropout(0.1)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

# 18 モデルコンパイル
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

#19  クラスウェイトの計算
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(trainY),
    y=trainY
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


#20  TensorBoardコールバックの設定
log_dir = "logs/fit/" + pd.to_datetime('now').strftime("%Y%m%d-%H%M%S") # ログの保存先
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#21 EarlyStoppingコールバックの設定 
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=0,
    restore_best_weights=True
)

# 22学習率スケジューラ（例：ステップ減衰）
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)

# 23 コールバックをリストに追加
lr_logger = LearningRateLogger()
callbacks = [lr_scheduler, lr_logger, early_stopping, tensorboard_callback] # TensorBoardコールバックを追加

# 24 モデルの学習
history = model.fit(trainX, trainY, epochs=100, batch_size=32, validation_split=0.2, class_weight=class_weight_dict, verbose=1,callbacks=callbacks  )

# 25 予測
testPredict = model.predict(testX)
testPredict = (testPredict > 0.5).astype(int)  # 0.5を閾値として分類

# 評価指標
from sklearn.metrics import classification_report

print(classification_report(testY, testPredict))


"""
tensorboardとか使う前に環境をアクティベートしてから!!!
conda activate env2 これがアクティベートするやつ

tensorboard --logdir logs/fit これがtensorborad起動用コード
tuna用でまた別で探索するハイパーパラメーターの組み合わせ等を準備する必要があるのでtunaのコードを参考に！

以下のほうがいいかも
cd C:\Users\sunak\anaconda3\
tensorboard --logdir C:\Users\sunak\anaconda3\logs\hparam_tuning

"""