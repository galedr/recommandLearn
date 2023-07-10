import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 讀取商品資料
with open('data/goods.json', 'r', encoding='utf-8') as f:
    goods_data = json.load(f)

# 提取商品標題作為特徵
titles = [data['title'] for data in goods_data]

# 提取其他欄位作為特徵
dc_c1 = [data['dc_c1'] for data in goods_data]
dc_c2 = [data['dc_c2'] for data in goods_data]
class1 = [data['class1'] for data in goods_data]
class2 = [data['class2'] for data in goods_data]
keyword = [data['keyword'] for data in goods_data]
cost = [data['sprice'] for data in goods_data]

# 計算特徵向量
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(titles)

# 欄位權重
dc_c1_weight = 2.0
dc_c2_weight = 1.0
keyword_weight = 2.5

# 產生目標商品的特徵向量
new_product_data = {
    'sn': '34199',
    'title': '《漫威蜘蛛人 2》中文收藏版（附贈預購特典）',
    'dc_c1': '2090',
    'dc_c2': '28',
    'class1': '28',
    'class2': '1',
    'keyword': '漫威蜘蛛人',
    'sprice': '1300'
}

new_product_vector = tfidf_vectorizer.transform([new_product_data['title']])

# 計算相似度
similarity_scores = cosine_similarity(new_product_vector, tfidf_matrix)
similarity_scores = similarity_scores.flatten()

# 根據相似度排序，取得推薦商品的索引
indices = similarity_scores.argsort()[::-1][:5]

# 根據索引取得推薦商品的標題
recommended_titles = [goods_data[idx]['title'] for idx in indices]

print(recommended_titles)
