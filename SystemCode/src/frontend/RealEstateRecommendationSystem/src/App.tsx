import { useState, useRef, useEffect } from 'react';
import { MessageBubble } from './components/MessageBubble';
import { ChatInput } from './components/ChatInput';
import { PropertyRecommendations } from './components/PropertyRecommendations';
import { InspirationGallery } from './components/InspirationGallery';
import { SuggestedQuestions } from './components/SuggestedQuestions';
import { createMessage, ChatMessage } from './utils/aiAgent';
import { generateResponse, initializeProperties, saveConversation, getConversationHistory } from './utils/api';
import { ScrollArea } from './components/ui/scroll-area';
import { Card } from './components/ui/card';
import { Button } from './components/ui/button';
import { Home, Bot, AlertCircle, RotateCcw } from 'lucide-react';
import { Alert, AlertDescription } from './components/ui/alert';

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sessionId] = useState(() => Date.now().toString() + Math.random().toString(36).substr(2, 9));
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Initialize the app
  useEffect(() => {
    const initializeApp = async () => {
      try {
        await initializeProperties();
        try {
          const { conversation } = await getConversationHistory(sessionId);
          if (conversation && conversation.length > 0) {
            setMessages(conversation);
          } else {
            const welcomeMessage = createMessage(
              "Hello! I'm your AI real estate agent. I'm here to help you find the perfect property. What type of home are you looking for today?",
              false
            );
            setMessages([welcomeMessage]);
          }
        } catch {
          const welcomeMessage = createMessage(
            "Hello! I'm your AI real estate agent. I'm here to help you find the perfect property. What type of home are you looking for today?",
            false
          );
          setMessages([welcomeMessage]);
        }
        setError(null);
      } catch (error) {
        console.error('Failed to initialize app:', error);
        setError('Failed to connect to the backend. Please refresh the page.');
      } finally {
        setIsInitializing(false);
      }
    };

    initializeApp();
  }, [sessionId]);

  // -----------------------------
  // ✅ 修正核心逻辑：使用 ai_response
  // -----------------------------
  const handleSendMessage = async (userMessage: string) => {
    try {
      const userMsg = createMessage(userMessage, true);
      setMessages(prev => [...prev, userMsg]);
      setIsLoading(true);
      setError(null);

      await saveConversation(sessionId, userMessage, true);

      // ✅ 修正这里：使用 ai_response
      const data = await generateResponse(userMessage);
      console.log("🧠 Backend returned:", data);

      const agentMessage = data.ai_response || data.message || "I'm here to help!";
      const agentMsg = createMessage(agentMessage, false);

      await saveConversation(sessionId, agentMessage, false);
      setMessages(prev => [...prev, agentMsg]);
    } catch (error) {
      console.error('Failed to send message:', error);
      setError('Failed to send message. Please try again.');
      const errorMsg = createMessage(
        "I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
        false
      );
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStartNewConversation = () => {
    setMessages([]);
    setError(null);
    const welcomeMessage = createMessage(
      "Hello! I'm your AI real estate agent. I'm here to help you find the perfect property. What type of home are you looking for today?",
      false
    );
    setMessages([welcomeMessage]);
  };

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]') as HTMLElement;
      if (scrollElement) scrollElement.scrollTop = scrollElement.scrollHeight;
    }
  }, [messages, isLoading]);

  if (isInitializing) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <Card className="p-8 text-center">
          <div className="flex items-center gap-2 mb-4">
            <div className="bg-primary text-primary-foreground p-2 rounded-lg">
              <Home className="w-5 h-5" />
            </div>
            <h1 className="font-semibold">AI Real Estate Agent</h1>
          </div>
          <div className="flex space-x-1 justify-center">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
          <p className="text-sm text-muted-foreground mt-2">Initializing your agent...</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-background">
      <div className="flex items-center gap-3 p-4 border-b bg-card">
        <Button variant="outline" size="sm" onClick={handleStartNewConversation} disabled={isLoading} className="flex items-center gap-2">
          <RotateCcw className="w-4 h-4" />
          Start
        </Button>
        <div className="flex items-center gap-2">
          <div className="bg-primary text-primary-foreground p-2 rounded-lg">
            <Home className="w-5 h-5" />
          </div>
          <div>
            <h1 className="font-semibold">AI Real Estate Agent</h1>
            <p className="text-sm text-muted-foreground">Find your perfect property</p>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2 text-green-600">
          <Bot className="w-4 h-4" />
          <span className="text-sm">Online</span>
        </div>
      </div>

      {error && (
        <Alert className="mx-4 mt-4" variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full p-4" ref={scrollAreaRef}>
          <div className="max-w-4xl mx-auto space-y-4 pb-4">
            {messages.map((message) => (
              <div key={message.id}>
                <MessageBubble
                  message={message.content}
                  isUser={message.isUser}
                  timestamp={message.timestamp}
                />
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3 mb-4">
                <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <Card className="p-4 flex items-center gap-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                  <span className="text-sm text-muted-foreground">AI is thinking...</span>
                </Card>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>

      <div className="border-t bg-background">
        <div className="max-w-4xl mx-auto">
          <SuggestedQuestions onQuestionClick={handleSendMessage} disabled={isLoading} />
        </div>
        <div className="border-t">
          <div className="max-w-4xl mx-auto">
            <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
          </div>
        </div>
      </div>
    </div>
  );
}
