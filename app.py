import streamlit as st
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_classic.memory import ConversationBufferWindowMemory

# 加载环境变量
load_dotenv()

# ==================== 1. 配置页面 ====================
st.set_page_config(page_title="AI心理伙伴", page_icon="🌿", layout="wide")
st.title("🌿 AI心理伙伴——情感陪伴与成长记录助手")
st.caption("🤝 每一次倾诉，都值得被认真记住 | 你的专属AI树洞")


# ==================== 2. 输出解释器（Pydantic模型）====================
class EmotionalResponse(BaseModel):
    """AI心理伙伴的结构化输出"""
    reply: str = Field(description="给用户的共情回复")
    emotional_state: str = Field(description="用户情绪：快乐/悲伤/焦虑/平静/愤怒/孤独")
    suggestion: str = Field(description="给用户的小建议")
    memory_note: Optional[str] = Field(default="", description="需要记住的用户信息")

    class Config:
        json_schema_extra = {
            "example": {
                "reply": "我能理解你的感受...",
                "emotional_state": "焦虑",
                "suggestion": "试试深呼吸5次",
                "memory_note": "用户最近在准备考试"
            }
        }

# ==================== 3. 输出解析器 ====================
parser = PydanticOutputParser(pydantic_object=EmotionalResponse)

# ==================== 4. 长期记忆管理（简化版）====================
class LongTermMemory:
    """长期记忆管理器 - 简化版"""

    def __init__(self, file_path: str = "data/long_term.json"):
        self.file_path = file_path
        self.memories = self._load()

    def _load(self) -> Dict:
        os.makedirs("data", exist_ok=True)
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"facts": []}

    def _save(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def add_fact(self, fact: str):
        if fact and fact not in self.memories["facts"]:
            self.memories["facts"].append(fact)
            self._save()

    def get_facts_string(self) -> str:
        if not self.memories["facts"]:
            return "暂无"
        return "，".join(self.memories["facts"])

    def clear(self):
        """清除所有长期记忆"""
        self.memories = {"facts": []}
        self._save()


# ==================== 5. 短期记忆 ====================
def create_short_term_memory(window_size: int = 6):
    """创建短期对话记忆"""
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=window_size,
        return_messages=True,
        input_key="input"
    )


# ==================== 6. RAG检索 ====================
class SimpleRAG:
    def __init__(self):
        self.file = "data/rag_memories.json"
        self.memories = self._load()

    def _load(self):
        if os.path.exists(self.file):
            with open(self.file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.file, 'w', encoding='utf-8') as f:
            json.dump(self.memories[-100:], f, ensure_ascii=False, indent=2)

    def add(self, text: str):
        if text and len(text) > 5:
            self.memories.append({
                "text": text,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            self._save()

    def search(self, query: str, top_k: int = 2) -> str:
        """简单的关键词检索"""
        if not self.memories:
            return "无"

        keywords = query.split()
        scored = []
        for mem in self.memories:
            score = sum(1 for kw in keywords if kw in mem["text"])
            if score > 0:
                scored.append((score, mem["text"]))

        scored.sort(reverse=True)
        results = [s[1] for s in scored[:top_k]]

        if results:
            return "\n".join([f"- {r[:100]}..." for r in results])
        return "无"

    def clear(self):
        self.memories = []
        self._save()


# ==================== 7. Agent工具 ====================
class AgentTools:
    def __init__(self):
        self.diary_file = "data/mood_diary.json"

    def write_diary(self, mood: str, content: str, score: int) -> str:
        """记录心情日记"""
        os.makedirs("data", exist_ok=True)
        entries = []
        if os.path.exists(self.diary_file):
            with open(self.diary_file, 'r', encoding='utf-8') as f:
                entries = json.load(f)

        entries.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mood": mood,
            "content": content,
            "score": score
        })

        with open(self.diary_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        return f"✅ 已记录：心情{score}/10分，{mood}"

    def get_mood_history(self, days: int = 7) -> str:
        """查询心情历史"""
        if not os.path.exists(self.diary_file):
            return "暂无心情记录"

        with open(self.diary_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)

        recent = entries[-days:] if len(entries) > days else entries
        if not recent:
            return "暂无记录"

        avg = sum(e['score'] for e in recent) / len(recent)
        result = f"📊 最近{len(recent)}天平均心情：{avg:.1f}/10\n"
        for e in recent:
            result += f"• {e['date'][:10]} | {e['score']}/10 | {e['mood']}\n"
        return result

    def get_tip(self, topic: str = "") -> str:
        """获取心理小知识"""
        tips = {
            "焦虑": "试试4-7-8呼吸法：吸气4秒，屏住7秒，呼气8秒",
            "抑郁": "完成一件小事，联系朋友，晒太阳15分钟",
            "睡眠": "睡前1小时不看手机，保持房间凉爽",
            "放松": "闭上眼睛深呼吸5次"
        }
        for k, v in tips.items():
            if k in topic:
                return v
        return "每天记录3件感恩的小事，能提升幸福感！"


# ==================== 8. 情绪检测函数 ====================
def detect_emotion(user_input: str) -> str:
    """快速检测用户情绪"""
    emotion_keywords = {
        "快乐": ["开心", "高兴", "快乐", "兴奋", "棒", "真好"],
        "悲伤": ["难过", "伤心", "痛苦", "失落"],
        "焦虑": ["焦虑", "紧张", "担心", "害怕"],
        "愤怒": ["生气", "愤怒", "恼火", "不爽"],
        "孤独": ["孤独", "寂寞", "一个人", "孤单"]
    }

    for emotion, keywords in emotion_keywords.items():
        for kw in keywords:
            if kw in user_input:
                return emotion
    return "平静"


# ==================== 9. 构建提示词模板 ====================
def create_prompt_template():
    """创建 LangChain 提示词模板"""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一位温暖、专业的AI心理伙伴"小暖"。

【关于用户的长期记忆】
{long_term_memory}

【检索到的相关记忆】
{rag_memories}

【你的角色】
- 倾听用户的心声，提供情感支持
- 根据用户情绪调整回复风格
- 记住用户提到的重要信息

【回复要求】
- 要温暖、共情、真诚
- 不要说教，不要评判
- 如果识别到负面情绪，提供具体的小建议

{format_instructions}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])


# ==================== 10. 构建Chain ====================
class PsychologicalChain:
    def __init__(self, api_key: str, temperature: float = 0.7, model: str = "deepseek-chat"):
        # 初始化LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model=model,
            temperature=temperature
        )

        # 初始化记忆
        self.short_term_memory = create_short_term_memory()
        self.long_term_memory = LongTermMemory()

        # 初始化RAG和工具
        self.rag = SimpleRAG()
        self.tools = AgentTools()

        # 创建提示词模板
        self.prompt = create_prompt_template()

        # 创建输出解析器
        self.parser = parser

        # 构建Chain
        self.chain = self._build_chain()

    def _prepare_inputs(self, inputs: Dict) -> Dict:
        """准备输入数据"""
        user_input = inputs.get("input", "")

        # 获取短期记忆
        short_mem = self.short_term_memory.load_memory_variables({})
        chat_history = short_mem.get("chat_history", [])

        # 获取长期记忆
        long_mem = self.long_term_memory.get_facts_string()

        # RAG检索
        rag_results = self.rag.search(user_input)
        rag_memories = rag_results if rag_results != "无" else "无"

        return {
            "input": user_input,
            "chat_history": chat_history,
            "long_term_memory": long_mem,
            "rag_memories": rag_memories,
            "format_instructions": self.parser.get_format_instructions()
        }

    def _parse_output(self, output: Any) -> Dict:
        """解析输出并保存记忆"""
        try:
            # 提取文本内容
            content = output.content if hasattr(output, 'content') else str(output)

            # 提取JSON（支持多种包裹格式）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            # 清理可能的空白字符
            content = content.strip()
            parsed = self.parser.parse(content)

            # 保存到短期记忆
            self.short_term_memory.save_context(
                {"input": parsed.memory_note or ""},
                {"output": parsed.reply}
            )

            # 保存到长期记忆
            if parsed.memory_note:
                self.long_term_memory.add_fact(parsed.memory_note)
                self.rag.add(parsed.memory_note)

            # 保存对话到RAG
            self.rag.add(f"对话：{parsed.reply[:100]}")

            return {
                "reply": parsed.reply,
                "emotional_state": parsed.emotional_state,
                "suggestion": parsed.suggestion,
                "memory_note": parsed.memory_note
            }
        except Exception as e:
            # ✅ 修复：解析失败时，只提取回复文本，不显示整个对象
            reply_text = output.content if hasattr(output, 'content') else str(output)
            # 如果内容显然不是JSON（长度过短或包含明显错误信息），直接返回文本
            return {
                "reply": reply_text,
                "emotional_state": "平静",
                "suggestion": "继续聊一聊吧",
                "memory_note": ""
            }

    def _build_chain(self):
        """构建 LangChain Runnable Sequence"""
        return (
                RunnablePassthrough().assign(**{"prepared": self._prepare_inputs})
                | RunnableLambda(lambda x: x["prepared"])
                | self.prompt
                | self.llm
                | RunnableLambda(self._parse_output)
        )

    def _check_tool_call(self, user_input: str) -> Optional[str]:
        """检查是否需要调用工具"""
        tool_keywords = {
            "记录心情": lambda: self.tools.write_diary("平静", user_input, 7),
            "心情分析": lambda: self.tools.get_mood_history(7),
            "心理知识": lambda: self.tools.get_tip(user_input),
            "小技巧": lambda: self.tools.get_tip(user_input),
        }

        for keyword, action in tool_keywords.items():
            if keyword in user_input:
                return action()
        return None

    def invoke(self, user_input: str) -> Dict:
        # 先检查工具调用
        tool_result = self._check_tool_call(user_input)

        # 检测情绪（用于前端显示）
        detected_emotion = detect_emotion(user_input)

        # 调用Chain
        result = self.chain.invoke({"input": user_input})

        # 合并工具结果
        if tool_result:
            result["reply"] = f"{result['reply']}\n\n{tool_result}"

        result["detected_emotion"] = detected_emotion

        return result

    def clear(self):
        """清除所有记忆"""
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        self.rag.clear()


# ==================== 11. 辅助函数 ====================
def export_chat_history(history):
    """导出对话历史"""
    import pandas as pd
    export_data = []
    for i, msg in enumerate(history):
        export_data.append({
            "序号": i + 1,
            "角色": msg["role"],
            "内容": msg["content"],
            "情绪": msg.get("emotion", ""),
        })
    df = pd.DataFrame(export_data)
    return df.to_csv(index=False).encode('utf-8')


# ==================== 12. 初始化Session State ====================
if "chain" not in st.session_state:
    st.session_state.chain = None
if "history" not in st.session_state:
    st.session_state.history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("DEEPSEEK_API_KEY", "")
if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

# ==================== 13. 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 配置")
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        key="api_key_input"
    )

    # 注意：这里重复定义了一次 temperature，需要删除下面重复的那一行
    temperature = st.slider(
        "回复创造性", 0.0, 1.0, 0.7, 0.1,
        key="temp_slider"
    )

    # 模型选择
    model_options = {
        "deepseek-chat": "DeepSeek Chat",
        "deepseek-coder": "DeepSeek Coder",
    }
    selected_model = st.selectbox(
        "选择模型",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )

    # ❌ 删除了重复的 temperature slider（原代码第456行附近有重复）

    if api_key and st.session_state.chain is None:
        with st.spinner("初始化 LangChain..."):
            st.session_state.chain = PsychologicalChain(api_key, temperature, selected_model)
        st.success("✅ Chain 已初始化")

    st.divider()

    if st.session_state.chain:
        st.header("📝 长期记忆")
        st.info(st.session_state.chain.long_term_memory.get_facts_string())

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清除记忆"):
                st.session_state.chain.clear()
                st.session_state.history = []
                st.rerun()
        with col2:
            if st.button("📊 心情分析"):
                result = st.session_state.chain.tools.get_mood_history()
                st.info(result)

    st.divider()

    if st.session_state.history:
        csv_data = export_chat_history(st.session_state.history)
        st.download_button(
            label="📥 导出对话",
            data=csv_data,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ==================== 14. 主界面 ====================
# 显示对话历史
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("emotion"):
            st.caption(f"💭 {msg['emotion']}")

# 快捷回复
st.markdown("### 快捷回复")
cols = st.columns(5)
quick_responses = [
    ("😊 分享开心事", "今天遇到一件特别开心的事！"),
    ("😢 寻求安慰", "最近心情不太好，想找人聊聊"),
    ("😰 缓解焦虑", "最近压力很大，总是焦虑"),
    ("📝 记录心情", "我想记录一下今天的心情"),
    ("💡 心理知识", "给我一些心理小知识")
]

for col, (label, text) in zip(cols, quick_responses):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.quick_prompt = text

st.markdown("---")

# 用户输入
if st.session_state.quick_prompt:
    prompt = st.session_state.quick_prompt
    st.session_state.quick_prompt = None
else:
    prompt = st.chat_input("说说你今天的心情...")

if prompt:
    if not api_key:
        st.error("请配置 API Key")
    elif st.session_state.chain is None:
        st.error("请等待初始化完成")
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("LangChain 思考中..."):
                try:
                    result = st.session_state.chain.invoke(prompt)

                    st.markdown(result["reply"])

                    emotion_icon = {"快乐": "😊", "悲伤": "😢", "焦虑": "😰",
                                    "平静": "😌", "愤怒": "😤", "孤独": "🫂"}.get(
                        result["emotional_state"], "💭")
                    st.caption(f"{emotion_icon} {result['emotional_state']}")
                    st.info(f"💡 {result['suggestion']}")

                    if result.get("memory_note"):
                        st.success(f"✨ 记住了：{result['memory_note']}")

                    st.session_state.history.append({
                        "role": "assistant",
                        "content": result["reply"],
                        "emotion": result["emotional_state"]
                    })

                except Exception as e:
                    st.error(f"错误：{e}")

st.divider()
st.caption("💚 基于 LangChain 构建的 AI 心理伙伴 | 提示词模板 | 输出解释器 | Chain链 | 记忆 | RAG | Agent")
