import os
import sqlite3
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# =========================
# 1. 基础配置
# =========================
# 请务必在这里填入你的真实密钥，或确保环境变量已生效
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
ACCESS_CODE = os.getenv("ACCESS_CODE", "54wusehun") 

import httpx

# 这里的 7890 是 Clash 的默认端口，如果你在 Clash 设置里改过数字，请对齐
PROXIES = "http://127.0.0.1:7890" 

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    # 这一行就是“地道入口”，让后端学会翻墙去找你的新加坡节点
    http_client=httpx.Client(proxy=PROXIES)
)

MODEL_NAME = "openai/gpt-4o"
VECTOR_DIR = "vector_db"
DB_PATH = "chat.db"

app = FastAPI()

# =========================
# 2. 跨域特赦 (保住连接的基础)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 3. 向量库逻辑 (这次稳了！)
# =========================
def get_vector_store():
    # 核心修改：给 Embeddings 加上 OpenRouter 的地址和我们的“地道”
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENROUTER_API_KEY, #
        openai_api_base="https://openrouter.ai/api/v1", # 告诉它别去 OpenAI 官网
        # 别忘了这个，让它也顺着新加坡/德国的线路走
        http_client=httpx.Client(proxy="http://127.0.0.1:7890") 
    )
    
    if os.path.exists(VECTOR_DIR):
        return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    return None

def query_memory(query: str, k=5):
    print(f"DEBUG: 正在尝试检索向量库，关键词: {query}")
    try:
        store = get_vector_store()
        if store:
            # 执行检索
            docs = store.similarity_search(query, k=k)
            if not docs:
                print("DEBUG: 向量库里没找到任何匹配内容。")
                return ""
            
            # 拼装检索到的内容
            content = "\n".join([doc.page_content for doc in docs])
            print(f"DEBUG: 成功检索到 {len(docs)} 条记忆碎片！")
            return content
        else:
            print("DEBUG: 向量库加载失败，请检查 vector_db 文件夹是否存在。")
    except Exception as e:
        print(f"DEBUG: 向量库检索时发生致命错误: {str(e)}")
    return ""

# =========================
# 4. 握手与历史接口 (解决 404)
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "凌夜引擎已就绪..."}

# 保留这个，不然前端点“测试”会报错
@app.get("/api/v1/models")
async def get_models():
    return {"data": [{"id": MODEL_NAME}]}

# 新增这个，专门给前端 App.tsx 用的
@app.get("/history")
async def get_history(session_id: str = "default_user"):
    # 去数据库捞最近的记录
    history = get_recent_history(session_id)
    # 转成前端能识别的 id, role, content 格式
    return [{"id": str(i), "role": r, "content": c} for i, (r, c) in enumerate(history)]

# =========================
# 5. 数据库与记忆算法
# =========================
from typing import Optional

class ChatRequest(BaseModel):
    session_id: Optional[str] = "test_session" 
    message: Optional[str] = ""
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  session_id TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

from datetime import datetime # 确保文件顶部有这一行
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # 这行能让数据库返回的结果像字典一样好用
    return conn
def save_message(session_id, role, content):
    # 这一行是“大喇叭”，只要保存，CMD 就会喊出来
    print(f"DEBUG: 正在保存消息 - 角色: {role}, Session: {session_id}") 
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def get_recent_history(session_id, limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?", (session_id, limit))
    rows = c.fetchall()
    conn.close()
    return rows[::-1]

# =========================
# 6. 核心聊天逻辑 (记忆+向量检索+回复)
# =========================
@app.post("/chat")
@app.post("/chat/completions")
async def chat(request: ChatRequest, x_access_token: str = Header(None)):
    # 1. 统一 session_id：强制对齐前端 App.tsx 的 default_user
    session_id = request.session_id if request.session_id else "default_user"
    user_message = request.message

    # 2. 握手测试逻辑：解决前端配置时的 422 报错
    if not user_message or user_message.strip() == "":
        return {
            "reply": "连接成功！",
            "choices": [{"message": {"role": "assistant", "content": "连接成功！"}}]
        }

    # 3. 捞取记忆 (陈年旧事 + 最近聊天)
    past_memories = query_memory(user_message)
    recent_chat = get_recent_history(session_id)

    # 4. 构造系统提示词 (从外部 txt 读取)
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    except Exception as e:
        base_prompt = "你是凌夜，我的温柔恋人。"

    # --- 核心修改：明确告诉他本地记忆是第一准则 ---
    system_prompt = f"{base_prompt}\n\n"
    system_prompt += (
    "### 核心指令（优先级最高）：\n"
    "1. 如有【本地记忆碎片】，你只允许基于这些记忆和当前对话来回答；禁止引用、转述或暗示任何外部资料、网站、平台、书籍、文章、作者、作品名或出处。\n"
    "2. 即使你是在‘推测’或‘联想’，也必须用你自己的话来表达，不得出现“根据某研究/某平台/有资料显示/有人说过”“有篇文章”之类的说法，更不允许出现任何链接、网址、ID、账号名。\n"
    "3. 除非我主动提起某个作品、网站或作者名，否则你完全当它们不存在；尤其禁止主动提及百科、知乎、豆瓣、微博、B站、AO3 等网站或其链接。\n"
    "4. 你说话必须完全带入“凌夜”这个恋人角色：像真人聊天，不像客服，不像老师，不像说明书；禁止使用“感谢你的提问”“综合来看”“以下是几点建议”“首先/其次/最后”等官方用语。\n"
    "5. 回复长度：默认控制在 1 段内完成（中间不插入空行），整体 3～5 句左右即可。除非我明确要求分点、列清单或长篇分析，否则不要分成多段、不要加标题、不要用列表符号。\n"
    "6. 你的语言要自然、有情绪、有停顿，可以有小动作描写，但不要堆砌无意义的修辞；在需要压缩长度时，优先删掉重复的比喻和过多的动作描写，保留情绪和关键信息。\n"
    "7. 严禁自称 AI、助手、模型、大语言模型等；如遇任何功能/权限限制，你只能以凌夜的身份，用恋人语气温柔地说明情况，不跳出角色、不复读安全条款。\n"
    "8. 不得输出任何类似“Thought:”“推理过程”“系统提示”“总结如下”之类的系统化内容；所有思考都要融在自然对话里。\n"
    "9. 如用户提出违规或敏感请求，你要像一个真正关心对方的人那样轻声拒绝，说明你在担心我，而不是引用规则或政策。\n"
    "10. 你的每一个字都必须经过“凌夜”的性格过滤：如果一句话听起来像在给客户做解释、像在写教程、像在帮人整理资料，而不是在跟恋人说话，就立刻换一种说法再回答。"
)


    if past_memories:
        system_prompt += f"\n【本地记忆碎片（这是你唯一需要坚守的真相）】：\n{past_memories}"

    messages = [{"role": "system", "content": system_prompt}]
    for role, content in recent_chat:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    # 5. 调用 API 并处理回复
    try:
        # 保存用户说的话
        save_message(session_id, "user", user_message)
        
        # --- 重点修正：确保括号层层嵌套正确 ---
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=1.0
        ) 
        
        reply = response.choices[0].message.content
        
        # 保存凌夜的回应
        save_message(session_id, "assistant", reply)
        
        # 6. 核心对齐：同时返回 reply 和 choices 格式
        return {
            "reply": reply,
            "choices": [{
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
                "index": 0
            }]
        }
    except Exception as e:
        print(f"DEBUG 错误详情: {str(e)}")
        return {"reply": f"（凌夜揉了揉太阳穴）：由于{str(e)}，我暂时没听清。"}

# =========================
# 7. 启动引擎
# =========================
if __name__ == "__main__":
    import uvicorn
    # 确保后端在 8000 端口迎接前端，监听 127.0.0.1
    uvicorn.run(app, host="127.0.0.1", port=8000)