// ==========================================
// File: src/utils/api.ts
// Purpose: Connect React frontend to local Python FastAPI backend
// ==========================================

const API_BASE = "http://localhost:8000"; // ⚙️ Python backend URL

// -----------------------------
// 通用请求封装
// -----------------------------
async function apiRequest(endpoint: string, options: RequestInit = {}) {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API Error (${response.status}):`, errorText);
      throw new Error(`API request failed: ${errorText}`);
    }

    const data = await response.json();
    console.log("✅ Backend response:", data); // 调试输出
    return data;
  } catch (error) {
    console.error("❌ API request error:", error);
    throw error;
  }
}

// -----------------------------
// API 接口定义（匹配 app.py 路由）
// -----------------------------

// 获取 AI 对话回复（调用 FastAPI /chat）
export async function generateResponse(userMessage: string) {
  return apiRequest("/chat", {
    method: "POST",
    body: JSON.stringify({
      user_input: userMessage, // ✅ 与 FastAPI ChatRequest 匹配
      thread_id: "1",
    }),
  });
}

// 初始化（可选）
export async function initializeProperties() {
  return { status: "ok" };
}

// 保存对话（本地模式略过）
export async function saveConversation(
  sessionId: string,
  message: string,
  isUser: boolean,
  properties?: any[]
) {
  console.log("💾 saveConversation skipped (local mode)");
  return { status: "saved" };
}

// 获取历史记录（本地模式返回空）
export async function getConversationHistory(sessionId: string) {
  return { conversation: [] };
}

// 健康检查
export async function healthCheck() {
  return apiRequest("/health");
}

// 房价预测接口
export async function predictHDBPrice(houseInfo: any) {
  return apiRequest("/predict_hdb_price", {
    method: "POST",
    body: JSON.stringify(houseInfo),
  });
}
