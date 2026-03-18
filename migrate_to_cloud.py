from neo4j import GraphDatabase

# ==========================================
# 1. 配置区 (请填入您的真实账密)
# ==========================================

# 本地数据库配置 (数据源)
LOCAL_URI = "bolt://localhost:7687"
LOCAL_USER = "neo4j"
LOCAL_PWD = "rthstyjujjj"  # 请替换

# 云端 AuraDB 配置 (目标地)
CLOUD_URI = "neo4j+s://af757b22.databases.neo4j.io"  # 请替换为您刚刚获取的 URI
CLOUD_USER = "af757b22"
CLOUD_PWD = "afWfha7hhBCDqQERMDbgVjlLAS-7z9tP84Vc7qsTYJc"  # 请替换

# ==========================================
# 2. Schema 定义 (基于您的医疗图谱规范)
# ==========================================
LABELS = ["Disease", "Symptom", "Drug", "Food", "Check", "Department", "Producer"]
RELATIONSHIPS = [
    "belongs_to", "common_drug", "do_eat", "drugs_of", "need_check", 
    "no_eat", "recommand_drug", "recommand_eat", "has_symptom", "acompany_with"
]

BATCH_SIZE = 1000  # 分批提交大小，防止云数据库因单次载入过多而超时

class DataMigrator:
    def __init__(self):
        print("正在连接数据库...")
        self.local_driver = GraphDatabase.driver(LOCAL_URI, auth=(LOCAL_USER, LOCAL_PWD))
        self.cloud_driver = GraphDatabase.driver(CLOUD_URI, auth=(CLOUD_USER, CLOUD_PWD))
        print("✅ 数据库连接成功！\n")

    def close(self):
        self.local_driver.close()
        self.cloud_driver.close()

    def migrate_nodes(self):
        print("========== 开始迁移节点 (Nodes) ==========")
        for label in LABELS:
            # 1. 从本地读取
            with self.local_driver.session() as local_session:
                result = local_session.run(f"MATCH (n:{label}) RETURN properties(n) AS props")
                nodes_data = [record["props"] for record in result]
            
            if not nodes_data:
                continue
                
            print(f"[{label}] 读取到 {len(nodes_data)} 个节点，正在写入云端...")
            
            # 2. 写入云端 (分批写入)
            with self.cloud_driver.session() as cloud_session:
                for i in range(0, len(nodes_data), BATCH_SIZE):
                    batch = nodes_data[i : i + BATCH_SIZE]
                    # 使用 UNWIND 批量创建节点
                    query = f"""
                    UNWIND $batch AS props
                    CREATE (n:{label})
                    SET n = props
                    """
                    cloud_session.run(query, batch=batch)
            print(f"  -> [{label}] 写入完成！")

    def create_indexes(self):
        print("\n========== 正在云端创建索引 (加速关系插入) ==========")
        with self.cloud_driver.session() as cloud_session:
            for label in LABELS:
                try:
                    cloud_session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)")
                except Exception as e:
                    pass # 如果索引已存在则忽略
        print("✅ 索引创建完毕！\n")

    def migrate_relationships(self):
        print("========== 开始迁移关系 (Relationships) ==========")
        for rel_type in RELATIONSHIPS:
            # 1. 从本地读取关系的源节点、目标节点及关系属性
            with self.local_driver.session() as local_session:
                query = f"""
                MATCH (n)-[r:{rel_type}]->(m)
                RETURN labels(n)[0] AS src_label, n.name AS src_name,
                       labels(m)[0] AS tgt_label, m.name AS tgt_name,
                       properties(r) AS props
                """
                result = local_session.run(query)
                rels_data = [record.data() for record in result]

            if not rels_data:
                continue
                
            print(f"[{rel_type}] 读取到 {len(rels_data)} 条关系，正在匹配并写入云端...")

            # 2. 写入云端
            with self.cloud_driver.session() as cloud_session:
                for i in range(0, len(rels_data), BATCH_SIZE):
                    batch = rels_data[i : i + BATCH_SIZE]
                    # 通过 name 匹配两端的节点，然后建立连接
                    write_query = f"""
                    UNWIND $batch AS row
                    MATCH (n {{name: row.src_name}}) WHERE row.src_label IN labels(n)
                    MATCH (m {{name: row.tgt_name}}) WHERE row.tgt_label IN labels(m)
                    CREATE (n)-[r:{rel_type}]->(m)
                    SET r = row.props
                    """
                    cloud_session.run(write_query, batch=batch)
            print(f"  -> [{rel_type}] 写入完成！")

if __name__ == "__main__":
    migrator = DataMigrator()
    try:
        # 第一步：把所有孤立的实体节点搬运过去
        migrator.migrate_nodes()
        
        # 第二步：给云端节点的 name 字段加上索引，不然下一步连线会非常慢
        migrator.create_indexes()
        
        # 第三步：按本地的连线方式，把云端的节点用“线”缝合起来
        migrator.migrate_relationships()
        
        print("\n🎉 恭喜！所有数据已成功迁移至 Neo4j AuraDB 云端！")
    except Exception as e:
        print(f"\n❌ 迁移过程中出现错误: {e}")
    finally:
        migrator.close()