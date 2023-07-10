import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 讀取商品資料
with open('data/goods.json', 'r', encoding='utf-8') as file:
    goods_data = json.load(file)

# 提取欄位資料
sn_list = []
title_list = []
dc_c1_list = []
dc_c2_list = []
class1_list = []
class2_list = []
keyword_list = []
cost_list = []
maker_list = []

for goods in goods_data:
    sn_list.append(goods['sn'])
    title_list.append(goods['title'])
    dc_c1_list.append(goods['dc_c1'])
    dc_c2_list.append(goods['dc_c2'])
    class1_list.append(goods['class1'])
    class2_list.append(goods['class2'])
    keyword_list.append(goods['keyword'])
    cost_list.append(goods['sprice'])
    maker_list.append(goods['maker'])

# 合併特徵欄位
features = ['title', 'dc_c1', 'dc_c2', 'class1', 'class2', 'keyword', 'sprice', 'maker']
corpus = [' '.join([str(goods[field]) for field in features]) for goods in goods_data]

# 使用 TfidfVectorizer 轉換特徵向量
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# 計算相似度矩陣
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

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
indices = similarity_scores.argsort()[0][-20:][::-1]

# 排除相似度較高的 title 商品
filtered_indices = []
for idx in indices:
    title_similar = False
    for filtered_idx in filtered_indices:
        title_similarity = cosine_similarity(tfidf_matrix[filtered_idx], tfidf_matrix[idx])[0][0]
        if title_similarity > 0.8:
            title_similar = True
            break
    if not title_similar:
        filtered_indices.append(idx)
    if len(filtered_indices) >= 10:
        break

# 取得推薦的十個商品
recommended_products = [goods_data[idx] for idx in filtered_indices]

# 輸出推薦的商品
for product in recommended_products:
    print(product['sn'], product['title'])
