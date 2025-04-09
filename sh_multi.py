import pandas as pd

# データの読み込み (すでに df が読み込まれていると仮定)
df = pd.read_csv("C:/Users/sunak/OneDrive/ドキュメント/心不全予測データセット.csv")
features = ['Age', 'Male', 'Female','RestingBP','Cholesterol','FastingBS','MaxHR',"ExerciseAngina","Oldpeak"] 
# 相関行列を計算
correlation_matrix = df[features].corr()

# 相関行列を表示
print(correlation_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

# 相関行列を計算 (上記と同じ)
correlation_matrix = df[features].corr()

# ヒートマップを描画
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()