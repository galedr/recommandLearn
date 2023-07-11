import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 讀取商品資料
with open('data/goods.json', 'r', encoding='utf-8') as file:
    goods_data = json.load(file)

# 提取欄位資料
features = ['title', 'dc_c1', 'dc_c2', 'class1', 'class2', 'keyword', 'sprice', 'maker']
data = [[goods[field] for field in features] for goods in goods_data]

# 轉換特徵向量
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([' '.join(item) for item in data])

# 定義特徵權重
feature_weights = [1, 1, 5, 1, 5, 1, 1, 1]  # 權重值請根據需求自行調整，轉為整數

# 計算加權後的特徵向量
weighted_tfidf_matrix = tfidf_matrix.copy()
for i, weight in enumerate(feature_weights):
    weighted_tfidf_matrix[:, i] = tfidf_matrix[:, i] * weight

# 定義新產品
new_product = {
    'sn': '34199',
    'title': '《漫威蜘蛛人 2》中文收藏版（附贈預購特典）',
    'dc_c1': '2090',
    'dc_c2': '28',
    'class1': '28',
    'class2': '1',
    'keyword': '漫威蜘蛛人',
    'sprice': 1392,
    'maker': 'SONY'
}

# 產生新產品特徵向量
new_product_vector = vectorizer.transform([' '.join([str(new_product[field]) for field in features])])

# 計算新產品與所有商品的相似度
similarity_scores = cosine_similarity(new_product_vector, weighted_tfidf_matrix)

# 取得相似度最高的十個商品索引
indices = similarity_scores.argsort()[0][-10:][::-1]

# 取得推薦的十個商品
recommended_products = [goods_data[idx] for idx in indices]

# 輸出推薦的商品
for product in recommended_products:
    print(product['sn'], product['title'])
