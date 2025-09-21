import json
import os
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import requests
import time


@dataclass
class ExtractedTriple:
    entities: List[Dict]
    relationships: List[Dict]
    attributes: List[Dict]


class DeepSeekClient:
    """DeepSeek API客户端"""

    def __init__(self, api_key, base_url="https://api.deepseek.com/v1", model="deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completions_create(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 1024,
                                max_retries: int = 3) -> Dict:
        """调用DeepSeek API - 带重试机制"""
        for attempt in range(max_retries):
            try:
                print(f"🔄 API调用尝试 {attempt + 1}/{max_retries}...")

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    timeout=60,  # 增加到60秒
                    proxies=None  # 禁用代理
                )
                response.raise_for_status()
                print(f"✅ API调用成功")
                return response.json()

            except requests.exceptions.Timeout:
                print(f"⏰ 第{attempt + 1}次调用超时")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 递增等待时间
                    print(f"   等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                continue

            except requests.exceptions.ConnectionError as e:
                print(f"🌐 第{attempt + 1}次连接错误: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"   等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                continue

            except requests.exceptions.HTTPError as e:
                print(f"❌ HTTP错误: {e}")
                if e.response.status_code == 429:  # 限流
                    print("   触发限流，等待更长时间...")
                    time.sleep(30)
                    continue
                else:
                    break

            except Exception as e:
                print(f"❌ 其他错误: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                continue

        # 所有重试都失败，返回空结果
        print(f"❌ API调用完全失败，返回空结果")
        return {"choices": [{"message": {"content": "{\"entities\": [], \"relationships\": []}"}}]}


class EntityExtractor:
    """实体关系抽取器"""

    def __init__(self, deepseek_api_key: str):
        self.deepseek = DeepSeekClient(deepseek_api_key)
        self.arduino_keywords = [
            'Arduino', 'LED', 'sensor', '传感器', 'pin', '引脚', 'GPIO',
            'voltage', '电压', 'current', '电流', 'resistor', '电阻', 'PWM',
            'digital', '数字', 'analog', '模拟', 'serial', '串口', 'I2C', 'SPI',
            'breadboard', '面包板', 'wire', '导线', 'ground', '接地', 'VCC', '5V', '3.3V'
        ]

    def _extract_page_number(self, filename: str) -> int:
        """从文件名中提取页面/幻灯片号码 - 支持多种格式"""
        # 🔧 修改：支持多种格式：slide_1, p1, page_1, _1, 等
        patterns = [
            r'slide_?(\d+)',  # slide_1, slide1
            r'p(\d+)',  # p1
            r'page_?(\d+)',  # page_1, page1
            r'_(\d+)',  # _1
            r'(\d+)'  # 纯数字
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return 0

    def load_multimodal_data(self, output_dir: str = "output") -> Dict:
        """加载多模态预处理的输出数据 - 兼容所有文件格式"""
        result = {
            'slides': [],
            'images': []
        }

        try:
            # 定义子目录路径
            text_dir = os.path.join(output_dir, "text")
            image_dir = os.path.join(output_dir, "images")

            # 1. 加载文本数据 (从text目录) - 🔧 移除格式限制
            if os.path.exists(text_dir):
                text_files = os.listdir(text_dir)
                # 🔧 修改：接受所有JSON文件，不限制格式
                slide_files = [f for f in text_files if
                               f.endswith('.json') and not f.endswith('_desc.json')]

                print(f"📄 找到文本文件: {len(slide_files)}个")
                for slide_file in slide_files:
                    print(f"   - {slide_file}")

                for slide_file in slide_files:
                    slide_path = os.path.join(text_dir, slide_file)
                    try:
                        with open(slide_path, 'r', encoding='utf-8') as f:
                            slide_data = json.load(f)

                        # 🔧 修改：使用新的页面号提取方法
                        slide_num = self._extract_page_number(slide_file)

                        result['slides'].append({
                            "slide_number": slide_num,
                            "content": slide_data,
                            "source_file": slide_file
                        })

                        print(f"✅ 加载文本文件: {slide_file} (页面 {slide_num})")

                    except Exception as e:
                        print(f"⚠️ 加载文本文件失败 {slide_file}: {e}")

            # 2. 加载图片数据 (从image目录) - 🔧 移除格式限制
            if os.path.exists(image_dir):
                image_files_list = os.listdir(image_dir)
                image_files = [f for f in image_files_list if
                               f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

                print(f"🖼️ 找到图片文件: {len(image_files)}个")

                for image_file in image_files:
                    # 查找对应的描述文件
                    base_name = os.path.splitext(image_file)[0]
                    desc_file = base_name + '_desc.json'
                    desc_path = os.path.join(image_dir, desc_file)

                    # 🔧 修改：使用新的页面号提取方法
                    slide_num = self._extract_page_number(image_file)

                    image_data = {
                        "image_path": os.path.join(image_dir, image_file),
                        "slide_number": slide_num,
                        "filename": image_file,
                        "descriptions": [],
                        "ocr_text": ""
                    }

                    # 如果有描述文件，加载描述信息
                    if os.path.exists(desc_path):
                        try:
                            with open(desc_path, 'r', encoding='utf-8') as f:
                                desc_data = json.load(f)
                                image_data["descriptions"] = desc_data.get("clip_descriptions", [])
                            print(f"✅ 加载图片描述: {desc_file}")
                        except Exception as e:
                            print(f"⚠️ 加载图片描述失败 {desc_file}: {e}")
                    else:
                        print(f"ℹ️ 未找到描述文件: {desc_file}")

                    result['images'].append(image_data)

            print(f"✅ 数据加载完成:")
            print(f"   - 文本文件: {len(result['slides'])}个")
            print(f"   - 图片文件: {len(result['images'])}个")

            # 按页面号排序
            result['slides'].sort(key=lambda x: x['slide_number'])
            result['images'].sort(key=lambda x: x['slide_number'])

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()

        return result

    def extract_entities_from_multimodal(self, multimodal_data: Dict) -> ExtractedTriple:
        """从多模态数据中抽取实体关系"""
        all_entities = []
        all_relationships = []
        all_attributes = []

        print("🔍 开始实体关系抽取...")

        # 1. 处理文本内容
        slides = multimodal_data.get('slides', [])
        print(f"📄 处理 {len(slides)} 个文本文件...")
        for i, slide in enumerate(slides):
            print(f"   处理文本 {i + 1}/{len(slides)}: {slide.get('source_file', '')}")
            slide_entities, slide_relations = self._extract_from_slide_text(slide)
            all_entities.extend(slide_entities)
            all_relationships.extend(slide_relations)

            # 🔧 增加随机延迟，避免频繁调用
            import random
            delay = random.uniform(1, 3)  # 1-3秒随机延迟
            print(f"   ⏳ 等待 {delay:.1f} 秒避免频繁调用...")
            time.sleep(delay)

        # 2. 处理图片内容
        images = multimodal_data.get('images', [])
        print(f"🖼️ 处理 {len(images)} 个图片文件...")
        for i, image_data in enumerate(images):
            print(f"   处理图片 {i + 1}/{len(images)}: {image_data.get('filename', '')}")
            img_entities = self._extract_from_image(image_data)
            all_entities.extend(img_entities)

        # 去重处理
        all_entities = self._deduplicate_entities(all_entities)
        all_relationships = self._deduplicate_relationships(all_relationships)

        print(f"✅ 实体关系抽取完成: {len(all_entities)}个实体, {len(all_relationships)}个关系")

        return ExtractedTriple(
            entities=all_entities,
            relationships=all_relationships,
            attributes=all_attributes
        )

    def _extract_from_slide_text(self, slide: Dict) -> Tuple[List[Dict], List[Dict]]:
        """从文本中抽取实体和关系"""
        slide_content = slide.get('content', {})
        slide_num = slide.get('slide_number', 0)

        # 提取文本内容 - 支持多种数据结构
        text_content = ""
        if isinstance(slide_content, dict):
            # 尝试多个可能的文本字段
            text_fields = ['text', 'content', 'raw_text', 'texts']
            for field in text_fields:
                if field in slide_content:
                    content = slide_content[field]
                    if isinstance(content, list):
                        text_content = ' '.join(str(item) for item in content)
                    else:
                        text_content = str(content)
                    break

            # 如果还没找到文本，尝试整个字典转字符串
            if not text_content:
                text_content = str(slide_content)
        else:
            text_content = str(slide_content)

        if not text_content or text_content.strip() == "" or text_content == "{}":
            print(f"   ⚠️ 页面 {slide_num} 无有效文本内容")
            return [], []

        # 🔧 分块处理长文本
        max_length = 2000  # 限制文本长度
        if len(text_content) > max_length:
            text_content = text_content[:max_length] + "..."
            print(f"   📝 文本过长，截取前{max_length}字符")

        # 构建提示词
        prompt = f"""
请从以下文档内容中抽取实体和关系。

内容：{text_content}

请识别以下类型的实体：
1. 硬件组件：Arduino板、传感器、LED、电阻、电容等
2. 技术概念：PWM、串口通信、数字信号、模拟信号等
3. 参数数值：电压值、电阻值、引脚号、频率等
4. 操作步骤：连接、编程、测试、调试等
5. 代码概念：函数、变量、库文件等
6. 一般概念：任何重要的名词、术语、概念

请识别实体间的关系：
- 组成关系：A包含B、A由B组成
- 连接关系：A连接到B、A接入B
- 控制关系：A控制B、A驱动B
- 参数关系：A的参数是B、A设置为B
- 功能关系：A用于B、A实现B
- 属于关系：A属于B、A是B的一种

严格按照以下JSON格式返回，不要添加任何其他内容：
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型", "description": "实体描述"}}
    ],
    "relationships": [
        {{"source": "源实体", "target": "目标实体", "relation": "关系类型"}}
    ]
}}
"""

        try:
            response = self.deepseek.chat_completions_create([
                {"role": "user", "content": prompt}
            ], max_retries=3)  # 使用重试机制

            content = response['choices'][0]['message']['content']

            # 提取JSON部分
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)

                # 添加页面信息
                entities = result.get('entities', [])
                for entity in entities:
                    entity['slide'] = slide_num
                    entity['source'] = 'text'

                relationships = result.get('relationships', [])
                for rel in relationships:
                    rel['slide'] = slide_num
                    rel['source'] = 'text'

                print(f"   ✅ 页面 {slide_num}: {len(entities)}个实体, {len(relationships)}个关系")
                return entities, relationships

        except Exception as e:
            print(f"   ⚠️ 页面 {slide_num} 实体抽取失败: {e}")

        return [], []

    def _extract_from_image(self, image_data: Dict) -> List[Dict]:
        """从图片数据中抽取实体"""
        entities = []
        slide_num = image_data.get('slide_number', 0)
        image_path = image_data.get('image_path', '')
        filename = image_data.get('filename', '')

        # 1. 基于图片描述抽取实体
        descriptions = image_data.get('descriptions', [])
        for desc_item in descriptions:
            desc_text = desc_item.get('description', '')
            confidence = desc_item.get('confidence', 0)

            if desc_text and confidence > 0.05:  # 置信度阈值
                entities.append({
                    'name': desc_text,
                    'type': 'image_concept',
                    'description': f'从图片描述中识别: {desc_text}',
                    'confidence': confidence,
                    'source': 'image_description',
                    'slide': slide_num,
                    'image_path': image_path,
                    'filename': filename
                })

        # 2. 基于OCR文本抽取实体（如果有OCR文本）
        ocr_text = image_data.get('ocr_text', '')
        if ocr_text:
            # Arduino关键词匹配
            for keyword in self.arduino_keywords:
                if keyword.lower() in ocr_text.lower():
                    entities.append({
                        'name': keyword,
                        'type': 'hardware_component',
                        'description': f'从图片OCR中识别的{keyword}',
                        'source': 'image_ocr',
                        'slide': slide_num,
                        'image_path': image_path,
                        'filename': filename
                    })

        # 3. 基于文件名抽取实体（如果文件名包含有用信息）
        if 'arduino' in filename.lower():
            entities.append({
                'name': 'Arduino',
                'type': 'hardware_platform',
                'description': '从文件名识别的Arduino平台',
                'source': 'filename',
                'slide': slide_num,
                'image_path': image_path,
                'filename': filename
            })

        return entities

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体去重"""
        seen = set()
        unique_entities = []

        for entity in entities:
            key = (entity['name'].lower(), entity['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """关系去重"""
        seen = set()
        unique_relationships = []

        for rel in relationships:
            key = (rel['source'].lower(), rel['target'].lower(), rel['relation'])
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)

        return unique_relationships


def extract_entities_from_output(output_dir: str, deepseek_api_key: str) -> ExtractedTriple:
    """从多模态输出中抽取实体关系的主函数"""
    extractor = EntityExtractor(deepseek_api_key)

    # 加载数据
    multimodal_data = extractor.load_multimodal_data(output_dir)

    # 抽取实体关系
    extracted_data = extractor.extract_entities_from_multimodal(multimodal_data)

    return extracted_data
from neo4j import GraphDatabase
import pandas as pd
from typing import Optional, Dict, List, Tuple
import json
from collections import defaultdict


class Neo4jKnowledgeGraph:
    """Neo4j知识图谱连接器"""

    def __init__(self, uri: str, user: str, password: str):
        """初始化Neo4j连接"""
        self.driver = None
        try:
            config = {
                "keep_alive": True,
                "max_connection_lifetime": 3600,
                "max_connection_pool_size": 100
            }
            self.driver = GraphDatabase.driver(uri, auth=(user, password), **config)

            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✅ Neo4j连接成功")

        except Exception as e:
            print(f"❌ Neo4j连接失败: {e}")
            raise

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()

    def clear_database(self):
        """清空数据库（谨慎使用）"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("🗑️ 数据库已清空")

    def create_document_node(self, doc_name: str, doc_type: str = "ppt", metadata: Dict = None):
        """创建文档节点"""
        with self.driver.session() as session:
            query = """
            MERGE (d:Document {name: $doc_name})
            SET d.type = $doc_type
            SET d.created_at = datetime()
            """

            if metadata:
                for key, value in metadata.items():
                    query += f"SET d.{key} = ${key} "

            params = {"doc_name": doc_name, "doc_type": doc_type}
            if metadata:
                params.update(metadata)

            session.run(query, params)
        print(f"📄 创建文档节点: {doc_name}")

    def create_entity_node(self, entity_name: str, entity_type: str, description: str = "",
                           metadata: Dict = None):
        """创建实体节点"""
        with self.driver.session() as session:
            query = """
            MERGE (e:Entity {name: $entity_name})
            SET e.type = $entity_type
            SET e.description = $description
            SET e.updated_at = datetime()
            """

            params = {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "description": description
            }

            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        query += f"SET e.{key} = ${key} "
                        params[key] = value

            session.run(query, params)

    def create_relationship(self, source_name: str, target_name: str, relation_type: str,
                            properties: Dict = None):
        """创建关系"""
        with self.driver.session() as session:
            # 确保源节点和目标节点存在
            session.run("""
                MERGE (s:Entity {name: $source_name})
                MERGE (t:Entity {name: $target_name})
            """, source_name=source_name, target_name=target_name)

            # 创建关系
            query = f"""
            MATCH (s:Entity {{name: $source_name}})
            MATCH (t:Entity {{name: $target_name}})
            MERGE (s)-[r:{relation_type}]->(t)
            SET r.created_at = datetime()
            """

            params = {"source_name": source_name, "target_name": target_name}

            if properties:
                for key, value in properties.items():
                    if isinstance(value, (str, int, float, bool)):
                        query += f"SET r.{key} = ${key} "
                        params[key] = value

            session.run(query, params)

    def save_extracted_data(self, extracted_data, ppt_name: str):
        """保存抽取的实体关系数据到Neo4j"""
        try:
            print(f"💾 开始保存数据到Neo4j: {ppt_name}")

            # 1. 创建PPT文档节点
            self.create_document_node(ppt_name, "ppt", {
                "total_entities": len(extracted_data.entities),
                "total_relationships": len(extracted_data.relationships)
            })

            # 2. 批量创建实体节点
            print(f"   创建 {len(extracted_data.entities)} 个实体节点...")
            entity_count = 0
            for entity in extracted_data.entities:
                try:
                    metadata = {k: v for k, v in entity.items()
                                if k not in ['name', 'type', 'description'] and
                                isinstance(v, (str, int, float, bool))}

                    self.create_entity_node(
                        entity['name'],
                        entity.get('type', 'unknown'),
                        entity.get('description', ''),
                        metadata
                    )

                    # 创建文档到实体的包含关系
                    self.create_relationship(ppt_name, entity['name'], "CONTAINS")
                    entity_count += 1

                except Exception as e:
                    print(f"     ⚠️ 创建实体失败 {entity['name']}: {e}")
                    continue

            # 3. 批量创建关系
            print(f"   创建 {len(extracted_data.relationships)} 个关系...")
            relation_count = 0
            for rel in extracted_data.relationships:
                try:
                    properties = {k: v for k, v in rel.items()
                                  if k not in ['source', 'target', 'relation'] and
                                  isinstance(v, (str, int, float, bool))}

                    self.create_relationship(
                        rel['source'],
                        rel['target'],
                        rel['relation'].upper().replace(' ', '_'),
                        properties
                    )
                    relation_count += 1

                except Exception as e:
                    print(f"     ⚠️ 创建关系失败 {rel['source']}->{rel['target']}: {e}")
                    continue

            print(f"✅ 数据保存完成:")
            print(f"   📄 文档: {ppt_name}")
            print(f"   🏷️  实体: {entity_count}/{len(extracted_data.entities)}")
            print(f"   🔗 关系: {relation_count}/{len(extracted_data.relationships)}")

            return {
                'document': ppt_name,
                'entities_saved': entity_count,
                'relationships_saved': relation_count,
                'success': True
            }

        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            return {
                'document': ppt_name,
                'entities_saved': 0,
                'relationships_saved': 0,
                'success': False,
                'error': str(e)
            }

    def query_entities(self, limit: int = 10) -> List[Dict]:
        """查询实体"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.name as name, e.type as type, e.description as description
                LIMIT $limit
            """, limit=limit)

            return [dict(record) for record in result]

    def query_relationships(self, limit: int = 10) -> List[Dict]:
        """查询关系"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Entity)-[r]->(t:Entity)
                RETURN s.name as source, type(r) as relation, t.name as target
                LIMIT $limit
            """, limit=limit)

            return [dict(record) for record in result]

    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        with self.driver.session() as session:
            # 节点统计
            node_result = session.run("MATCH (n) RETURN count(n) as total_nodes")
            total_nodes = node_result.single()['total_nodes']

            # 关系统计
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as total_relationships")
            total_relationships = rel_result.single()['total_relationships']

            # 实体类型统计
            entity_type_result = session.run("""
                MATCH (e:Entity)
                RETURN e.type as entity_type, count(e) as count
                ORDER BY count DESC
            """)
            entity_types = [dict(record) for record in entity_type_result]

            # 文档统计
            doc_result = session.run("MATCH (d:Document) RETURN count(d) as total_documents")
            total_documents = doc_result.single()['total_documents']

            return {
                'total_nodes': total_nodes,
                'total_relationships': total_relationships,
                'total_documents': total_documents,
                'entity_types': entity_types
            }

    def search_entities_by_name(self, name_pattern: str, limit: int = 10) -> List[Dict]:
        """按名称搜索实体"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($pattern)
                RETURN e.name as name, e.type as type, e.description as description
                LIMIT $limit
            """, pattern=name_pattern, limit=limit)

            return [dict(record) for record in result]

    def get_entity_neighbors(self, entity_name: str, depth: int = 1) -> Dict:
        """获取实体的邻居节点"""
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH path = (e:Entity {{name: $entity_name}})-[*1..{depth}]-(neighbor)
                RETURN neighbor.name as name, neighbor.type as type, 
                       neighbor.description as description
                LIMIT 20
            """, entity_name=entity_name)

            neighbors = [dict(record) for record in result]

            # 获取相关关系
            rel_result = session.run("""
                MATCH (e:Entity {name: $entity_name})-[r]-(neighbor)
                RETURN neighbor.name as neighbor, type(r) as relation,
                       CASE WHEN startNode(r).name = $entity_name 
                            THEN 'outgoing' ELSE 'incoming' END as direction
            """, entity_name=entity_name)

            relationships = [dict(record) for record in rel_result]

            return {
                'entity': entity_name,
                'neighbors': neighbors,
                'relationships': relationships
            }


def save_to_neo4j(extracted_data, ppt_name: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    """保存数据到Neo4j的主函数"""
    kg = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)

    try:
        result = kg.save_extracted_data(extracted_data, ppt_name)

        # 显示统计信息
        stats = kg.get_statistics()
        print(f"\n📊 数据库统计信息:")
        print(f"   📄 文档总数: {stats['total_documents']}")
        print(f"   🏷️  节点总数: {stats['total_nodes']}")
        print(f"   🔗 关系总数: {stats['total_relationships']}")

        # 显示实体类型分布
        print(f"\n🏷️  实体类型分布:")
        for entity_type in stats['entity_types'][:5]:
            print(f"   - {entity_type['entity_type']}: {entity_type['count']}个")

        return result

    finally:
        kg.close()


def output_to_neo4j(output_dir: str,
                    deepseek_api_key: str,
                    neo4j_uri: str,
                    neo4j_user: str,
                    neo4j_password: str,
                    ppt_name: str = "Arduino课程PPT",
                    clear_database: bool = False,
                    show_examples: bool = True) -> Dict:
    """
    从output目录提取实体关系并保存到Neo4j的主函数

    Args:
        output_dir: 输出目录路径，包含text和images子目录
        deepseek_api_key: DeepSeek API密钥
        neo4j_uri: Neo4j数据库URI，如 "bolt://localhost:7687"
        neo4j_user: Neo4j用户名
        neo4j_password: Neo4j密码
        ppt_name: PPT文档名称，默认"Arduino课程PPT"
        clear_database: 是否清空数据库，默认False
        show_examples: 是否显示查询示例，默认True

    Returns:
        Dict: 包含处理结果的字典
    """

    result = {
        'success': False,
        'entities_extracted': 0,
        'relationships_extracted': 0,
        'entities_saved': 0,
        'relationships_saved': 0,
        'error': None
    }

    kg = None

    try:
        print("🚀 开始从output到Neo4j的完整流程...")
        print(f"   📁 输出目录: {output_dir}")
        print(f"   🗄️  Neo4j: {neo4j_uri}")
        print(f"   📄 文档名称: {ppt_name}")

        # 1. 实体关系抽取
        print("\n🔍 步骤1: 实体关系抽取...")
        extracted_data = extract_entities_from_output(output_dir, deepseek_api_key)

        result['entities_extracted'] = len(extracted_data.entities)
        result['relationships_extracted'] = len(extracted_data.relationships)

        print(f"✅ 抽取完成:")
        print(f"   🏷️  实体数量: {result['entities_extracted']}")
        print(f"   🔗 关系数量: {result['relationships_extracted']}")

        if result['entities_extracted'] == 0 and result['relationships_extracted'] == 0:
            print("⚠️ 未抽取到任何实体和关系，请检查输入数据")
            return result

        # 2. 连接Neo4j
        print(f"\n🗄️  步骤2: 连接Neo4j数据库...")
        kg = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)

        # 3. 可选：清空数据库
        if clear_database:
            print("🗑️ 清空数据库...")
            kg.clear_database()

        # 4. 保存数据到Neo4j
        print(f"\n💾 步骤3: 保存数据到Neo4j...")
        save_result = kg.save_extracted_data(extracted_data, ppt_name)

        result['entities_saved'] = save_result['entities_saved']
        result['relationships_saved'] = save_result['relationships_saved']
        result['success'] = save_result['success']

        if not save_result['success']:
            result['error'] = save_result.get('error', '保存失败')
            return result

        # 5. 显示统计信息
        print(f"\n📊 步骤4: 数据库统计信息...")
        stats = kg.get_statistics()
        print(f"   📄 文档总数: {stats['total_documents']}")
        print(f"   🏷️  节点总数: {stats['total_nodes']}")
        print(f"   🔗 关系总数: {stats['total_relationships']}")

        # 显示实体类型分布
        if stats['entity_types']:
            print(f"\n🏷️  实体类型分布:")
            for entity_type in stats['entity_types'][:5]:
                print(f"   - {entity_type['entity_type']}: {entity_type['count']}个")

        # 6. 可选：显示查询示例
        if show_examples:
            print(f"\n🔍 步骤5: 数据库查询示例...")

            # 查询实体示例
            entities = kg.query_entities(5)
            if entities:
                print(f"\n🏷️  实体示例:")
                for i, entity in enumerate(entities):
                    desc = entity.get('description', '')[:50] + (
                        '...' if len(entity.get('description', '')) > 50 else '')
                    print(f"   {i + 1}. {entity['name']} ({entity['type']}) - {desc}")

            # 查询关系示例
            relationships = kg.query_relationships(5)
            if relationships:
                print(f"\n🔗 关系示例:")
                for i, rel in enumerate(relationships):
                    print(f"   {i + 1}. {rel['source']} --{rel['relation']}--> {rel['target']}")

        # 7. 成功完成
        print(f"\n🎉 流程完成!")
        print(f"   ✅ 实体抽取: {result['entities_extracted']}个")
        print(f"   ✅ 关系抽取: {result['relationships_extracted']}个")
        print(f"   ✅ 实体保存: {result['entities_saved']}/{result['entities_extracted']}")
        print(f"   ✅ 关系保存: {result['relationships_saved']}/{result['relationships_extracted']}")

        result['success'] = True

    except Exception as e:
        print(f"❌ 流程失败: {e}")
        result['error'] = str(e)
        result['success'] = False

        import traceback
        traceback.print_exc()

    finally:
        # 确保关闭Neo4j连接
        if kg:
            kg.close()

    return result

from pathlib import Path
# 使用示例
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent  # 获取根目录
    input_file_path = BASE_DIR / "input"
    output_file_path = BASE_DIR / "output"
    # 调用示例
    result = output_to_neo4j(
        output_dir = str(output_file_path),
        deepseek_api_key="sk-c28ec338b39e4552b9e6bded47466442",
        neo4j_uri="bolt://101.132.130.25:7687",
        neo4j_user="neo4j",
        neo4j_password="wangshuxvan@1",
        ppt_name="Arduino课程PPT",
        clear_database=False,  # 是否清空数据库
        show_examples=True  # 是否显示查询示例
    )

    # 检查结果
    if result['success']:
        print(f"\n✅ 处理成功!")
        print(f"   实体: {result['entities_saved']}/{result['entities_extracted']}")
        print(f"   关系: {result['relationships_saved']}/{result['relationships_extracted']}")
    else:
        print(f"\n❌ 处理失败: {result.get('error', '未知错误')}")
