import google.generativeai as genai
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置 API key
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="AIzaSyB-qsnPlpBVdVWqvXDUwIHbxYhhkklfezY")

def init_chat():
    """初始化聊天模型"""
    # 获取 Gemini-Pro 模型
    model = genai.GenerativeModel('gemini-2.5-flash')
    # 创建聊天会话
    chat = model.start_chat(history=[])
    return chat

def chat_with_gemini(chat, user_input):
    """与 Gemini 进行对话"""
    try:
        response = chat.send_message(user_input)
        return response.text
    except Exception as e:
        return f"发生错误: {str(e)}"

def main():
    # 初始化聊天
    chat = init_chat()
    
    print("开始与 Gemini 对话 (输入 'quit' 退出)")
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() == 'quit':
            break
            
        response = chat_with_gemini(chat, user_input)
        print(f"\nGemini: {response}")

if __name__ == "__main__":
    main()