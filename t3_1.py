import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 讀取商品資料
with open('data/goods.json', 'r', encoding='utf-8') as file:
    goods_data = json.load(file)

# 提取欄位資料
features = ['title', 'dc_c1', 'dc_c2', 'class1', 'class2', 'keyword', 'sprice', 'maker']
data = [[goods[field] for field in features] for goods in goods_data]

# 定義特徵權重
feature_weights = [1, 1, 5, 1, 1, 1, 1, 1]  # 權重值請根據需求自行調整

# 計算特徵向量
vectors = []
for item in data:
    weighted_item = []
    for i, value in enumerate(item):
        if isinstance(value, str):  # 如果值是字串，直接加入權重調整後的特徵向量
            weighted_item.append(value)
        else:  # 如果值是數字，進行權重調整後再加入特徵向量
            weighted_item.append(value * feature_weights[i])
    vectors.append(' '.join([str(val) for val in weighted_item]))  # 將特徵向量轉換為空格分隔的字符串

# 使用 TfidfVectorizer 轉換特徵向量
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(vectors)

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
similarity_scores = cosine_similarity(new_product_vector, tfidf_matrix)

# 取得相似度最高的十個商品索引
indices = similarity_scores.argsort()[0][-10:][::-1]

# 取得推薦的十個商品
recommended_products = [goods_data[idx] for idx in indices]

# 輸出推薦的商品
for product in recommended_products:
    print(product['sn'], product['title'])
