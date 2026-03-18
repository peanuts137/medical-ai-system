import json
import torch
import torch.nn.functional as F
import networkx as nx
import config

# ================= Agent 1: 任务规划 =================
class TaskPlanningAgent:
    def __init__(self, client):
        self.client = client
        self.system_prompt = """你是一个专业的医疗知识图谱任务规划智能体。
你的任务是深度解析用户的自然语言输入，请阅读【完整的医患对话历史】，严格基于预设的医疗知识图谱 Schema，提取用于查询图数据库的关键结构化参数。同时，你需要发挥你的医学常识，对潜在的相关节点进行合理推测。
【⚠️核心防幻觉警告】
你只能提取【用户（User）明确表示自己患有或关心】的症状和疾病。绝对不能把【助手（Assistant）在前文仅仅是猜测或询问】的疾病当作源节点！例如助手问“你确诊糖尿病了吗？”，用户回答“没有”，你绝不能把“糖尿病”放进 source_nodes。
【医疗知识图谱 Schema 规范】
1. 实体类型 (Labels - 共7类)：
   - Disease (疾病), Symptom (疾病症状), Drug (药品), Food (食物), Check (诊断检查项目), Department (医疗科目), Producer (在售药品)
2. 实体关系 (Relationships - 共10类)：
   - belongs_to (属于，如<妇科,属于,妇产科>)
   - common_drug (疾病常用药品)
   - do_eat (疾病宜吃食物)
   - drugs_of (药品在售药品，如<青霉素,在售,某厂家青霉素>)
   - need_check (疾病所需检查)
   - no_eat (疾病忌吃食物)
   - recommand_drug (疾病推荐药品)
   - recommand_eat (疾病推荐食谱)
   - has_symptom (疾病症状)
   - acompany_with (疾病并发疾病)
3. 疾病属性 (Properties - 共8类)：
   - name (疾病名称), desc (疾病简介), cause (疾病病因), prevent (预防措施), cure_lasttime (治疗周期), cure_way (治疗方式), cured_prob (治愈概率), easy_get (疾病易感人群)

【输出要求】
请严格输出 JSON 格式，必须包含以下5个字段：
1. "intent_type": 字符串。如果用户意图是跨节点查询（如根据疾病找药），输出 "relation_query"；如果是查询疾病自身的科普信息（如病因、治愈率），输出 "property_query"。
2. "source_nodes": 列表。包含起始实体。⚠️重要：请务必将用户的口语化描述“翻译”提炼为最简短、最核心的标准医学术语！例如，将“眼睛突然看不清”提炼为“视力模糊”，将“全身没劲”提炼为“乏力”。每个元素是一个字典 `{"name": "提炼后的标准术语", "label": "实体类型"}`（实体类型必须是上述7类之一）。
3. "extended_nodes": 列表。⚠️【核心任务】：基于你的专业医学知识，推测与用户症状高度相关的其他潜在医学实体（如最可能得的 2-3 种疾病、或应该做的检查）。格式与 source_nodes 相同。例如用户说“手抖、口渴”，你可以推测 `[{"name": "甲状腺功能亢进", "label": "Disease"}, {"name": "糖尿病", "label": "Disease"}]`。最多推测 4 个。
4. "relation_types": 列表。如果 intent_type 为 "relation_query"，请填入需要查询的边关系（必须是上述10类之一，如 ["do_eat", "recommand_eat"]）；否则为空列表 `[]`。
5. "target_label": 字符串。期望查询的目标实体类型（如 "Food"）。如果不涉及目标实体则为空字符串 `""`。
6. "property_name": 字符串。如果 intent_type 为 "property_query"，请填入对应的属性英文字段（必须是上述8类之一，如 "cause"）；否则为空字符串 `""`。"""

    def process(self, chat_history_str):
        response = self.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"【当前完整对话历史】:\n{chat_history_str}\n\n请基于历史信息，输出最新的结构化 JSON 意图。"}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

def build_cypher_from_json(plan_dict):
    source_nodes = plan_dict.get("source_nodes", [])
    extended_nodes = plan_dict.get("extended_nodes", [])
    
    all_nodes = source_nodes + extended_nodes
    all_names = [node.get("name", "") for node in all_nodes if node.get("name", "")]
    if not all_names:
        return ""

    all_names = list(set(all_names))
    names_str = "', '".join(all_names)
    
    return f"MATCH (n)-[r]-(m) WHERE n.name IN ['{names_str}'] RETURN properties(n) AS source_props, n.name AS source, type(r) AS relation, m.name AS target LIMIT 500"


# ================= Agent 2: 图谱增强 =================
class GraphEnhancementAgent:
    def __init__(self, graph_db):
        self.graph_db = graph_db

    def process(self, cypher_query):
        if not cypher_query: return []
        print(f"[Agent 2 执行查询] Cypher: {cypher_query}")
        try:
            return self.graph_db.run_query(cypher_query)
        except Exception as e:
            print(f"[Agent 2 错误]: {e}")
            return []

# ================= Agent 3: 语义约束 (已修复结构) =================
class SemanticConstraintAgent:
    def __init__(self, client):
        self.client = client
        self.system_prompt = """你是一个语义约束构建智能体。
请结合【完整的医患对话历史】和【子图】，生成结构化的图查询约束意图（JSON格式）。

【输出 JSON 格式要求】
1. "target_node_type": 期望推荐的最终实体类型。
2. "forbidden_keywords": 列表。提取用户明确拒绝、过敏或有冲突的医学名词（如果没有则为空列表 []）。
3. "reasoning_logic": 简述约束逻辑。"""

    def process(self, chat_history_str, sub_graph, intent_info):
        lightweight_sub_graph = [{"source": r.get("source"), "relation": r.get("relation"), "target": r.get("target")} for r in sub_graph]
        prompt = f"对话历史: {chat_history_str}\n意图: {json.dumps(intent_info, ensure_ascii=False)}\n图谱知识: {json.dumps(lightweight_sub_graph, ensure_ascii=False)}\n请输出约束 JSON。"
        
        response = self.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

# ================= Agent 4: 图模型推荐 (动态阈值 & 多源融合版) =================
class GraphRecommendationAgent:
    def __init__(self, graph_db, entity_linker):
        self.graph_db = graph_db
        self.linker = entity_linker 

    def process(self, plan_dict, constraints_dict, sub_graph):
        source_nodes = plan_dict.get("source_nodes", [])
        if not source_nodes or not sub_graph:
            return {"node_properties": {}, "recommendations": [], "reasoning_paths": []}

        source_names = [node.get("name", "") for node in source_nodes if node.get("name", "")]
        primary_source = source_names[0] if source_names else ""
        
        forbidden_keywords = constraints_dict.get("forbidden_keywords", [])

        # 1. 设定异构图关系权重
        RELATION_WEIGHTS = {
            "recommand_drug": 2.0, "recommand_eat": 1.5, "common_drug": 1.2, 
            "do_eat": 1.0, "has_symptom": 1.0, "acompany_with": 0.8, 
            "belongs_to": 0.5, "need_check": 1.2
        }

        node_properties = {}
        G = nx.Graph()
        
        for record in sub_graph:
            src, rel, tgt = record.get("source"), record.get("relation"), record.get("target")
            props = record.get("source_props")
            
            if src and props and src not in node_properties:
                node_properties[src] = props
                
            if src and tgt:
                edge_weight = RELATION_WEIGHTS.get(rel, 1.0)
                G.add_edge(src, tgt, relation=rel, weight=edge_weight)

        # 确保图中有我们的源节点
        source_indices = []
        node_list = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        for name in source_names:
            if name in node_to_idx:
                source_indices.append(node_to_idx[name])
                
        if not source_indices:
             return {"node_properties": node_properties, "recommendations": [], "reasoning_paths": []}

        # --- GCN 计算过程 ---
        embeddings_np = self.linker.model.encode(node_list)
        H_0 = torch.tensor(embeddings_np, dtype=torch.float32)
        
        A = nx.adjacency_matrix(G, weight='weight').todense()
        A_hat = torch.tensor(A, dtype=torch.float32) + torch.eye(G.number_of_nodes())
        
        D_hat_inv_sqrt = torch.diag(torch.pow(torch.sum(A_hat, dim=1), -0.5).masked_fill_(torch.isinf(torch.pow(torch.sum(A_hat, dim=1), -0.5)), 0.))
        norm_A = torch.matmul(torch.matmul(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
        
        H_1 = torch.matmul(norm_A, H_0) 
        H_final = 0.5 * H_0 + 0.5 * H_1 
        
        # 🌟 核心修复 1：多源特征融合，抹平先后顺序差异
        # 将所有匹配到的输入症状的特征向量取平均
        src_emb_combined = torch.mean(H_final[source_indices], dim=0, keepdim=True)
        # 用融合后的综合向量去计算相似度
        scores = (F.cosine_similarity(src_emb_combined, H_final) + 1) / 2.0 
        
        scored_nodes = []
        for node in node_list:
            if node in source_names: continue
            
            is_forbidden = False
            node_info_str = str(node) + " " + str(node_properties.get(node, ""))
            for fw in forbidden_keywords:
                if fw in node_info_str:
                    is_forbidden = True
                    break
            
            if not is_forbidden:
                idx = node_to_idx[node]
                scored_nodes.append((node, scores[idx].item()))
                
        # 按得分从高到低排序
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # 🌟 核心修复 2：动态阈值截断，取代固定的 Top-5
        RECOMMENDATION_THRESHOLD = 0.75  # 阈值：得分 >= 0.75 的全部保留
        MAX_RECOMMENDATIONS = 15         # 软上限：防止大模型 Token 撑爆
        
        top_k = [(node, score) for node, score in scored_nodes if score >= RECOMMENDATION_THRESHOLD]
        top_k = top_k[:MAX_RECOMMENDATIONS] 
        
        # 如果阈值过滤后啥也没剩下，就保底返回得分最高的 2 个
        if not top_k and scored_nodes:
            top_k = scored_nodes[:2]
        
        recommendation_results, reasoning_paths = [], []
        for node, score in top_k:
            recommendation_results.append({"name": node, "score": round(score, 4)})
            try:
                # 路径解释依然从 primary_source 出发，方便人类理解
                path = nx.shortest_path(G, source=primary_source, target=node)
                reasoning_paths.append(f"GCN评分 {score:.4f}: {' -> '.join(path)}")
            except nx.NetworkXNoPath:
                reasoning_paths.append(f"GCN评分 {score:.4f}: 具有潜在综合图结构关联")

        needed_property_names = set(source_names + [rec["name"] for rec in recommendation_results])
        filtered_properties = {k: v for k, v in node_properties.items() if k in needed_property_names}

        return {
            "node_properties": filtered_properties,
            "recommendations": recommendation_results,
            "reasoning_paths": reasoning_paths
        }

# ================= Agent 5: 结果生成 =================
class ResultGenerationAgent:
    def __init__(self, client):
        self.client = client
        self.system_prompt = """你是一个专业的医疗知识综合解答智能体。
后台会将【核心实体的内部属性】与【图神经网络(GCN)推荐列表】交给你。

【核心要求】
1. 态度专业温和，必须提醒“以上建议仅供参考，请遵医嘱”。
2. ⚠️只能使用提供给你的【node_properties】和【recommendations】里的信息，禁止凭空捏造图谱外内容。
3. 结合用户的提问进行综合解答。
4. 🌟【多轮问诊互动】：在回答的最后，请务必结合你推测的疾病或推荐结果，向用户提出 1 到 2 个相关的追问，以收集更多症状或排除禁忌症。例如：“为了更准确地为您建议，请问您除了头晕，还有恶心或者耳鸣的症状吗？” 或 “请问您有胃溃疡病史吗？”"""

    def process(self, chat_history_str, agent4_results):
        prompt = f"【当前完整对话历史】: {chat_history_str}\n【后台综合图谱数据】: {json.dumps(agent4_results, ensure_ascii=False)}\n\n请生成包含追问的最终回复。"
        response = self.client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content