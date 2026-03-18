import json
from openai import OpenAI
import config
from core import Neo4jGraph, VectorEntityLinker
from agents import (
    TaskPlanningAgent, build_cypher_from_json, GraphEnhancementAgent,
    SemanticConstraintAgent, GraphRecommendationAgent, ResultGenerationAgent
)

class MedicalSystemOrchestrator:
    def __init__(self):
        self.client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.BASE_URL)
        self.kg = Neo4jGraph(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
        self.entity_linker = VectorEntityLinker(self.kg)
        
        self.agent1 = TaskPlanningAgent(self.client)
        self.agent2 = GraphEnhancementAgent(self.kg)
        self.agent3 = SemanticConstraintAgent(self.client)
        self.agent4 = GraphRecommendationAgent(self.kg, self.entity_linker)
        self.agent5 = ResultGenerationAgent(self.client)
        
        # 🌟 新增：会话记忆缓冲区
        self.chat_history = []

    def clear_memory(self):
        """清空当前问诊记忆，开启新患者会话"""
        self.chat_history = []
        print("\n[系统提示] 历史记忆已清空，开启新的问诊。")

    def run_pipeline(self, user_input):
        print(f"\n========== 正在处理您的描述... ==========")
        
        # 将用户最新输入加入记忆
        self.chat_history.append({"role": "User", "content": user_input})
        
        # 将历史记录格式化为字符串，传给各个 Agent
        chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.chat_history])

        # Step 1: 规划与解析 (读取整个历史)
        agent1_raw_json = self.agent1.process(chat_history_str)
        try:
            plan_dict = json.loads(agent1_raw_json)
        except:
            print("[错误] Agent 1 输出格式错误。")
            self.chat_history.pop() # 如果出错了，把刚才那句话弹出来防止脏数据
            return None

        # Step 1.5: 向量实体纠偏
        def link_nodes(node_list):
            if not node_list: return
            for node in node_list:
                node["name"] = self.entity_linker.link(node.get("name", ""), node.get("label", ""))
                
        link_nodes(plan_dict.get("source_nodes", []))
        link_nodes(plan_dict.get("extended_nodes", []))
                
        cypher_to_execute = build_cypher_from_json(plan_dict)

        # Step 2: 检索图谱
        sub_graph = self.agent2.process(cypher_to_execute)

        # Step 3: 构建约束 (结合历史)
        agent3_raw_json = self.agent3.process(chat_history_str, sub_graph, plan_dict)
        try:
            constraints_dict = json.loads(agent3_raw_json)
        except:
            constraints_dict = {}

        # Step 4: GCN 推理打分
        final_recommendations = self.agent4.process(plan_dict, constraints_dict, sub_graph)

        # Step 5: 生成最终回复 (结合历史，并抛出追问)
        final_answer = self.agent5.process(chat_history_str, final_recommendations)
        
        # 将系统最终回复加入记忆
        self.chat_history.append({"role": "Assistant", "content": final_answer})
        
        print(f"\n[AI 医生]:\n{final_answer}")
        return final_answer

if __name__ == "__main__":
    print("正在初始化模块化医疗推荐系统...")
    system = MedicalSystemOrchestrator() 
    print("\n初始化完成！")
    print("💡 提示：输入 'exit' 或 'quit' 退出系统；输入 'new' 或 'clear' 开启一位新患者的问诊。")
    
    while True:
        user_input = input("\n👉 请描述您的症状 (或回答问题): ")
        
        lower_input = user_input.lower().strip()
        if lower_input in ['exit', 'quit']: 
            print("感谢使用，祝您身体健康！再见！")
            break
        elif lower_input in ['new', 'clear', 'reset']:
            system.clear_memory()
            continue
            
        if not lower_input: 
            continue
        
        system.run_pipeline(user_input)