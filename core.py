import os
import pickle
import jieba
import numpy as np
from neo4j import GraphDatabase  # <--- 就是漏了这一句！
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import config

# ================= 数据库连接与查询类 =================
class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def run_query(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]
        
    def get_all_entity_names(self, label):
        query = f"MATCH (n:{label}) RETURN n.name AS name"
        try:
            results = self.run_query(query)
            # 过滤掉空的 name，并去重
            return list(set([res['name'] for res in results if res.get('name')]))
        except Exception as e:
            print(f"[数据库错误] 拉取 {label} 词库失败: {e}")
            return []

# ================= 向量实体链接与缓存类 (混合检索版) =================
class VectorEntityLinker:
    def __init__(self, graph_db, force_update=False):
        """
        初始化实体链接器。
        """
        self.cache_file = "entity_cache.pkl"
        print("\n[系统初始化] 正在加载本地轻量级中文医疗向量模型...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        # 核心逻辑：判断缓存是否存在
        if os.path.exists(self.cache_file) and not force_update:
            print("[系统初始化] 检测到本地词库缓存，正在极速加载...")
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.standard_symptoms = cache_data['symptoms']
                self.standard_diseases = cache_data['diseases']
                self.symptom_embeddings = cache_data['symptom_embs']
                self.disease_embeddings = cache_data['disease_embs']
                self.symptom_bm25 = cache_data['symptom_bm25']
                self.disease_bm25 = cache_data['disease_bm25']
            print(f"[系统初始化] 缓存加载完毕！共 {len(self.standard_symptoms)} 个症状, {len(self.standard_diseases)} 个疾病。")
            
        else:
            print("[系统初始化] 无可用缓存或强制更新。正在从 Neo4j 数据库拉取标准实体词库...")
            self.standard_symptoms = graph_db.get_all_entity_names("Symptom")
            self.standard_diseases = graph_db.get_all_entity_names("Disease")
            
            print(f"[系统初始化] 成功拉取 {len(self.standard_symptoms)} 个症状, {len(self.standard_diseases)} 个疾病。正在计算词库向量...")
            self.symptom_embeddings = self.model.encode(self.standard_symptoms) if self.standard_symptoms else []
            self.disease_embeddings = self.model.encode(self.standard_diseases) if self.standard_diseases else []
            
            print("[系统初始化] 正在构建 BM25 字面索引...")
            symptom_tokenized = [list(jieba.cut(name)) for name in self.standard_symptoms]
            disease_tokenized = [list(jieba.cut(name)) for name in self.standard_diseases]
            self.symptom_bm25 = BM25Okapi(symptom_tokenized) if symptom_tokenized else None
            self.disease_bm25 = BM25Okapi(disease_tokenized) if disease_tokenized else None
            
            # 将计算好的数据打包存入本地文件
            cache_data = {
                'symptoms': self.standard_symptoms,
                'diseases': self.standard_diseases,
                'symptom_embs': self.symptom_embeddings,
                'disease_embs': self.disease_embeddings,
                'symptom_bm25': self.symptom_bm25,
                'disease_bm25': self.disease_bm25
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print("[系统初始化] 计算完毕，并已成功保存至本地缓存！下次启动将实现秒开。")

        print("[系统初始化] 实体链接器准备完毕！\n")

    def link(self, raw_mention, entity_label, threshold=config.SIMILARITY_THRESHOLD):
        if entity_label == "Symptom" and self.standard_symptoms:
            standard_names = self.standard_symptoms
            standard_embs = self.symptom_embeddings
            bm25_model = self.symptom_bm25
        elif entity_label == "Disease" and self.standard_diseases:
            standard_names = self.standard_diseases
            standard_embs = self.disease_embeddings
            bm25_model = self.disease_bm25
        else:
            return raw_mention 

        # 1. 计算向量语义得分 (0~1)
        mention_emb = self.model.encode([raw_mention])
        vec_scores = cosine_similarity(mention_emb, standard_embs)[0]
        
        # 2. 计算 BM25 字面匹配得分
        tokenized_mention = list(jieba.cut(raw_mention))
        bm25_scores = bm25_model.get_scores(tokenized_mention)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 0
        if max_bm25 > 0:
            bm25_scores = bm25_scores / max_bm25 

        # 3. 混合打分 (Hybrid Scoring)
        alpha_vec = 0.4
        beta_bm25 = 0.6
        final_scores = (alpha_vec * vec_scores) + (beta_bm25 * bm25_scores)
        
        best_idx = np.argmax(final_scores)
        best_score = final_scores[best_idx]
        
        if best_score >= threshold:
            print(f"  -> [混合纠偏] '{raw_mention}' 映射为 '{standard_names[best_idx]}' (综合分数: {best_score:.2f})")
            return standard_names[best_idx]
        else:
            return raw_mention