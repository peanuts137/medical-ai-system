from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 导入您刚刚完善的主控调度流
from main import MedicalSystemOrchestrator

# 初始化 FastAPI 应用
app = FastAPI(title="AI 智能问诊系统后端 API")

# ⚠️ 极其重要：配置跨域资源共享 (CORS)
# 允许您本地双击打开的 HTML 文件访问这个后端的接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境下允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化我们在 main.py 里写好的医疗推荐系统
# 由于有缓存机制，这里启动会非常快
print("正在加载 AI 医疗核心引擎...")
ai_system = MedicalSystemOrchestrator()

# 定义前端传过来的数据格式
class ChatRequest(BaseModel):
    message: str

# 定义与前端通信的 API 路由
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 接收前端用户的输入
        user_input = request.message
        
        # 调用核心引擎的 run_pipeline 获取最终 AI 的回答
        # 这中间经历了 Agent 1~5 的所有图谱检索和 GCN 推理
        final_answer = ai_system.run_pipeline(user_input)
        
        if not final_answer:
            final_answer = "抱歉，系统处理您的描述时遇到了点问题，请换种说法试试。"
            
        # 将结果打包成 JSON 返回给前端
        return {"reply": final_answer}
    
    except Exception as e:
        print(f"[后端错误]: {e}")
        return {"reply": f"服务器内部错误，请检查控制台日志。"}

# 启动服务器的入口
if __name__ == "__main__":
    print("🚀 后端服务已启动！正在监听 http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)