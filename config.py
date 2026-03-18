import os

# OpenAI / DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-c5dcc68ef1b54eeab265041536518e5c")
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

# Neo4j 数据库配置
NEO4J_URI = "neo4j+s://af757b22.databases.neo4j.io"
NEO4J_USER = "af757b22"
NEO4J_PASSWORD = "afWfha7hhBCDqQERMDbgVjlLAS-7z9tP84Vc7qsTYJc"

# 向量模型配置
EMBEDDING_MODEL = 'shibing624/text2vec-base-chinese'
SIMILARITY_THRESHOLD = 0.55