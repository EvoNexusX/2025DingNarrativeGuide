import os
import re
import csv
import time
import openai
import pandas as pd
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_fixed

# 配置 OpenAI API
openai.api_base = 'https://xiaoai.plus/v1'  # 请根据实际API接口修改
openai.api_key = 'sk-34dTtiEg5s0gK5m2Fc716eEeEe764b63BeDcEd9a6182E5B9'  # 替换为您的实际API密钥

# 使用 tenacity 库实现重试逻辑
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def fetch_gpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-32k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error fetching GPT response: {e}")
        raise

# 评分提示模板
SCORING_PROMPT = """
# Task Instructions
You are a very strict and severe script evaluation expert, and you need to score the script excerpt based on the following four dimensions, using a 1-10 scale. The scoring should strictly follow the standards.Also, consider the length of the content. If the script excerpt is very short, points should be deducted accordingly.

# Attraction Information
{attraction_info}

#Scoring Criteria:

1. Plot Coherence
*Event Logic
0 points: Major logical errors; plot is chaotic with no causal relationships.
3 points: Obvious logical gaps between events; cause-effect relationships are missing or poorly defined.
5 points: Generally logical, but some events lack clear cause-effect links, causing minor confusion.
7 points: Logical structure is clear, but occasional forced connections reduce immersion.
10 points: All events are rigorously connected through natural cause-effect chains (e.g., historical events at one attraction directly trigger actions at the next).
*Attraction Relevance
0 points: Attractions are completely disconnected.
3 points: Weak historical/cultural links between attractions (e.g., superficial mentions).
5 points: Moderate thematic links, but missing multi-dimensional connections.
7 points: Strong multi-dimensional links (historical + cultural + geographical).
10 points: Seamless integration of attractions into a unified narrative (e.g., chain reactions of historical events across attractions).
*Transition Smoothness
0 points: Transitions are abrupt or nonexistent.
3 points: Transitions are disjointed; audience struggles to follow.
5 points: Transitions rely heavily on dialogue explanations
7 points: Smooth transitions but occasional awkward phrasing.
10 points: Transitions use knowledge graph relationships (e.g., "From the Yellow Crane Tower, we follow the Tang Dynasty trade route to the Grand Canal").

2. Character Interaction
*Dialogue Authenticity
0 points: Dialogue mismatches character backgrounds.
3 points: Dialogue feels robotic or generic.
5 points: Dialogue mostly fits characters but lacks uniqueness.
7 points: Dialogue reflects character identities and cultural context.
10 points: Dialogue organically blends character traits with attraction-specific cultural nuances.
*Cultural-Driven Actions
0 points: Character actions ignore cultural/historical context.
3 points: Superficial cultural references in actions.
5 points: Actions loosely tied to attraction themes.
7 points: Actions directly motivated by attraction history (e.g., writing letters after visiting a historical site).
10 points: Actions form narrative metaphors (e.g., using porcelain fragility to symbolize diplomatic tensions).

3. Time and Space Coherence
*Spatiotemporal Corridor
0 points: Illogical time/space jumps break immersion.
3 points: Time/space shifts lack clear motivation.
5 points: Basic adherence to historical timelines.
7 points: Time/space transitions align with attraction relationships (e.g., Ming Palace → Versailles via parallel timelines).
10 points: Transitions use knowledge graph coordinates for hyper-realistic consistency.
*Route Rationality
0 points: Path contradicts historical/geographical constraints.
3 points: Path is plausible but lacks depth.
5 points: Path follows basic historical logic.
7 points: Path reflects period-specific travel methods (e.g., horse carriages in Tang Dynasty).
10 points: Path optimization integrates multi-attraction causality (e.g., trade routes influencing plot progression).

4. Cultural Fit
*Cultural Depth
0 points: Attractions are mere backdrops.
3 points: Weak thematic links (e.g., mentioning history without impact).
5 points: Moderate integration (e.g., historical facts guide minor decisions).
7 points: Deep cultural interplay (e.g., poetry from an attraction shapes character arcs).
10 points: Attractions drive plot twists and thematic metaphors (e.g., Berlin Wall graffiti revealing Cold War tensions).
*Multi-Dimensional Linkage
0 points: No cross-attraction connections.
3 points: Single-dimension links (e.g., historical only).
5 points: Two-dimensional links (e.g., historical + geographical).
7 points: Multi-dimensional synergy (e.g., historical events + cultural symbols + geographic paths).
10 points: Holistic narrative network (e.g., An Lushan Rebellion → linked attraction construction → character motivations).

# Script Input                                                                                                  
{script_content}

# Output Requirements
Please strictly output in the following format:
1. **Plot Coherence**
   - **Event Logic**: [score]
   - **Attraction Relevance**: [score]
   - **Transition Smoothness**: [score]

2. **Character Interaction**
   - **Dialogue Authenticity**: [score]
   - **Cultural-Driven Actions**: [score]

3. **Time and Space Coherence**
   - **Spatiotemporal Corridor**: [score]
   - **Route Rationality**: [score]


4. **Cultural Fit**
   - **Cultural Depth**: [score]
   - **Multi-Dimensional Linkage**: [score]

"""

# 中英文通用的列名映射
COLUMN_MAPPING = {
    'Attraction Name': 'Attraction Name',
    '景点名称': 'Attraction Name',
    'Historical Background': 'Historical Background',
    '历史背景': 'Historical Background',
    'Cultural Significance': 'Cultural Significance',
    '文化特色': 'Cultural Significance',
    'Historical Stories': 'Historical Stories',
    '历史故事': 'Historical Stories',
    'Main Attractions': 'Main Attractions',
    '主要景点': 'Main Attractions',
    'Geographical Location': 'Geographical Location',
    '地理位置': 'Geographical Location',
}

# 定义各维度权重配置（可在此处调整权重比例）
DIMENSION_WEIGHTS = {
    "Plot Coherence": {
        "Event Logic": 0.4,
        "Attraction Relevance": 0.4,
        "Transition Smoothness": 0.2
    },
    "Character Interaction": {
        "Dialogue Authenticity": 0.3,
        "Culture-Driven Actions": 0.4,
        "Metaphorical Dialogue": 0.3
    },
    "Time and Space Coherence": {
        "Spatiotemporal Corridor": 0.6,
        "Route Rationality": 0.4
    },
    "Cultural Fit": {
        "Cultural Depth": 0.5,
        "Multi-Dimensional Linkage": 0.5
    }
}

def load_attractions_data(csv_path: str) -> Dict[str, Dict]:
    """从 CSV 文件加载景点数据，支持中英文列名"""
    attractions = {}
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            print(f"Column names in {csv_path}: {reader.fieldnames}")  # 调试信息
            if not reader.fieldnames:
                print(f"警告：CSV 文件为空：{csv_path}")
                return attractions
            
            # 去除列名中的 BOM 字符并转换为小写
            fieldnames = [col.strip('\ufeff').strip() for col in reader.fieldnames]
            reader.fieldnames = fieldnames
            
            # 动态匹配中英文列名
            column_mapping = {}
            for col in fieldnames:
                for key in COLUMN_MAPPING:
                    if col.lower() == key.lower() or col.lower() == COLUMN_MAPPING[key].lower():
                        column_mapping[col] = COLUMN_MAPPING[key]
            
            # 检查必要的列是否存在
            required_columns = {'Attraction Name', 'Historical Background', 
                               'Cultural Significance', 'Historical Stories', 
                               'Main Attractions', 'Geographical Location'}
            missing_columns = required_columns - set(column_mapping.values())
            if missing_columns:
                print(f"警告：{csv_path} 中缺少列：{missing_columns}")
                print(f"可用列：{fieldnames}")
                return attractions
            
            for row in reader:
                attraction_name = row.get(column_mapping.get('Attraction Name', ''), '').strip()
                if not attraction_name:
                    continue  # 跳过空景点名称
                
                attraction_info = {
                    'Historical Background': row.get(column_mapping.get('Historical Background', ''), '').strip(),
                    'Cultural Significance': row.get(column_mapping.get('Cultural Significance', ''), '').strip(),
                    'Historical Stories': row.get(column_mapping.get('Historical Stories', ''), '').strip(),
                    'Main Attractions': row.get(column_mapping.get('Main Attractions', ''), '').strip(),
                    'Geographical Location': row.get(column_mapping.get('Geographical Location', ''), '').strip(),
                }
                attractions[attraction_name] = attraction_info
    except Exception as e:
        print(f"加载 {csv_path} 时出错：{str(e)}")
    return attractions

def format_attraction_info(attraction_data: Dict) -> str:
    """格式化景点信息以用于提示"""
    return (
        f"Historical Background: {attraction_data.get('Historical Background', '')}\n"
        f"Cultural Significance: {attraction_data.get('Cultural Significance', '')}\n"
        f"Historical Stories: {attraction_data.get('Historical Stories', '')}\n"
        f"Main Attractions: {attraction_data.get('Main Attractions', '')}\n"
        f"Geographical Location: {attraction_data.get('Geographical Location', '')}"
    )

# 重试逻辑函数
def retry_request(func, max_retries=3, delay=2, *args, **kwargs):
    """如果函数调用失败，则尝试重试"""
    attempt = 0
    while attempt < max_retries:
        try:
            return func(*args, **kwargs)  # 尝试 API 调用
        except openai.error.OpenAIError as e:
            attempt += 1
            print(f"尝试 {attempt}/{max_retries} 失败，错误：{e}。{delay} 秒后重试...")
            time.sleep(delay)  # 等待后重试
        except Exception as e:
            print(f"意外错误：{e}")
            break
    return None  # 如果所有重试都失败，则返回 None
#防止短文本评分虚高
def adjust_weight_by_length(script_length):
    if script_length < 1500:
        return 0.7  # 短文本所有维度权重×0.7
    elif 1500 <= script_length < 7000:
        return 1.0  # 中文本权重不变
    else:
        return 1.2  # 长文本奖励分
# 修改 evaluate_script_with_gpt 函数以不切分剧本
def evaluate_script_with_gpt(script_content: str, attraction_info: str) -> Dict[str, float]:
    """使用 GPT 评估剧本"""
    scores = {
        "Plot Coherence": 0.0,
        "Character Interaction": 0.0,
        "Time and Space Coherence": 0.0,
        "Cultural Fit": 0.0,
        "Total Score": 0.0
    }

    # 获取剧本长度
    script_length = len(script_content)

    # 获取长度调整权重
    length_weight = adjust_weight_by_length(script_length)

    try:
        response = retry_request(openai.ChatCompletion.create,
            model="gpt-4-32k",
            messages=[ 
                {"role": "system", "content": "You are a strict script evaluation expert."},
                {"role": "user", "content": SCORING_PROMPT.format(attraction_info=attraction_info, script_content=script_content)}
            ],
            temperature=0.1
        )
        
        if response is None:  # 如果所有重试都失败，则返回空分数
            print("在重试后未能从 GPT-4 获取响应")
            return scores
        
        result_text = response['choices'][0]['message']['content']
        
        weighted_scores = {
            "Plot Coherence": [],
            "Character Interaction": [],
            "Time and Space Coherence": [],
            "Cultural Fit": []
        }
        
        current_dimension = None
        for line in result_text.split('\n'):
            line = line.strip()
            
            # 检测主维度标题（增强格式兼容性）
            if re.match(r'^\d+\.', line):  # 匹配任何数字开头的行
                current_dimension = line.split('.')[1].strip().strip("*: ")
                continue
            
            if current_dimension and ":" in line:
                key_part, value_part = line.split(":", 1)
                sub_dimension = key_part.strip().strip("-* ")
                raw_value = value_part.split("(")[0].strip()
                
                # 获取该小维度的原始权重
                weights = DIMENSION_WEIGHTS.get(current_dimension, {})
                if not weights:
                    continue
                
                if _is_valid_score(raw_value):
                    score = float(raw_value)
                    # 乘以小维度的权重和长度调整后的权重
                    weighted_score = score * weights.get(sub_dimension, 0) * length_weight
                    
                    if current_dimension in weighted_scores:
                        weighted_scores[current_dimension].append(weighted_score)
        
        # 计算主维度分数（加权平均）
        for dimension in scores:
            if dimension == "Total Score":
                continue
                
            dimension_scores = weighted_scores.get(dimension, [])
            if dimension_scores:
                total_weight = sum(DIMENSION_WEIGHTS[dimension].values())
                # 保持原有归一化计算逻辑
                scores[dimension] = sum(dimension_scores) / total_weight
        
        # 修改总分计算为各维度相加（原始分数范围：每个维度0-10分）
        valid_dimensions = [scores[d] for d in scores if d != "Total Score"]
        scores["Total Score"] = sum(valid_dimensions)  # 直接相加

    except Exception as e:
        print(f"Evaluation error: {str(e)}")
    
    return scores



def _is_valid_score(value: str) -> bool:
    """检查分数是否有效"""
    try:
        score = float(value)
        return 1 <= score <= 10
    except ValueError:
        return False

def process_scripts(input_dir: str, attractions_data_dir: str) -> list:
    """处理所有剧本文件"""
    data = []
    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue

        parts = filename.split('_')
        if len(parts) < 2:  # 至少包含城市名称和任何后缀
            print(f"文件名格式无效：{filename}，跳过")
            continue
        
        city_name = parts[0]
        
        # 加载对应的城市景点数据
        attractions_csv = os.path.join(attractions_data_dir, f"{city_name}_data.csv")
        if not os.path.exists(attractions_csv):
            print(f"警告：没有 {city_name} 的数据文件")
            continue
        
        attractions_data = load_attractions_data(attractions_csv)
        if not attractions_data:
            print(f"警告：{city_name} 数据中未找到景点")
            continue
        
        # 读取剧本内容
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # 提取提到的景点（与之前相同）
        mentioned_attractions = []
        for attraction_name in attractions_data.keys():
            pattern = re.compile(rf'\b{re.escape(attraction_name)}\b', re.IGNORECASE)
            if pattern.search(script_content):
                mentioned_attractions.append(attraction_name)
        
        if not mentioned_attractions:
            print(f"{filename} 中未找到提到的景点，使用所有景点")
            mentioned_attractions = list(attractions_data.keys())
        
        # 合并提到的景点信息
        combined_attraction_info = "\n\n".join([ 
            format_attraction_info(attractions_data[name]) 
            for name in mentioned_attractions
        ])
        
        # 直接传递整个剧本进行评估
        scores = evaluate_script_with_gpt(script_content, combined_attraction_info)
        print(f"Scores for {filename}: {scores}")  # 调试信息
        
        # 存储结果
        data.append({
            "Script Filename": filename,
            "City": city_name,
            **scores
        })
    return data

def save_to_csv(data: list, output_path: str):
    """将结果保存到 CSV 文件"""
    if not data:
        print("没有数据可保存")
        return
    
    columns = ["Script Filename", "City", 
               "Plot Coherence", "Character Interaction",
               "Time and Space Coherence", "Cultural Fit", "Total Score"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存到 {output_path}")

if __name__ == "__main__":
    input_dir = "scripts"  # 剧本文件夹路径
    attractions_data_dir = "citydata"  # 景点数据文件夹路径
    output_csv = "script_scores.csv"  # 输出文件路径
    
    data = process_scripts(input_dir, attractions_data_dir)
    save_to_csv(data, output_csv)