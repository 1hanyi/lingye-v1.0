import React, { useState, useEffect, useRef, useMemo } from 'react';
import Header from './components/Header';
import MessageList from './components/MessageList';
import InputBar from './components/InputBar';
import ApiConfigModal from './components/ApiConfigModal';
import SystemPromptModal from './components/SystemPromptModal';
import InterfaceConfigModal from './components/InterfaceConfigModal';
import LogsModal from './components/LogsModal';
import { OpenAIIcon } from './components/Icons'; 
import { Message, ApiConfig, InterfaceConfig, LogEntry } from './types';
import { exportChatToHtml } from './utils'; // 确保你有这个 utils 文件

const DEFAULT_CONFIG: ApiConfig = {
  baseUrl: 'http://167.172.84.137:8000',
  apiKey: '',
  model: 'gpt-4o',
  temperature: 0.7,
  top_p: 1.0,
  contextLimit: 40,
  enableAutoSummary: false
};

const DEFAULT_INTERFACE_CONFIG: InterfaceConfig = {
    centerHeader: false,
    showAvatar: false,
    pureBlack: false
};

const App: React.FC = () => {
  const [messagesMap, setMessagesMap] = useState<Record<string, Message>>({});
  const [currentLeafId, setCurrentLeafId] = useState<string | null>(null);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [isSystemPromptOpen, setIsSystemPromptOpen] = useState(false);
  const [isInterfaceConfigOpen, setIsInterfaceConfigOpen] = useState(false);
  const [isLogsOpen, setIsLogsOpen] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [apiConfig, setApiConfig] = useState<ApiConfig>(DEFAULT_CONFIG);
  const [interfaceConfig, setInterfaceConfig] = useState<InterfaceConfig>(DEFAULT_INTERFACE_CONFIG);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [editTargetId, setEditTargetId] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  // --- 【新增】刷新网页自动同步历史记录 ---
  // --- 刷新网页自动同步历史记录 ---
  useEffect(() => {
    const fetchHistory = async () => {
        try {
            const url = `${apiConfig.baseUrl.replace(/\/$/, '')}/history?session_id=default_user`;
            const response = await fetch(url, {
                headers: { 'x-access-token': apiConfig.apiKey }
            });
            if (response.ok) {
                const historyData = await response.json();
                
                // 把后端传来的列表转成前端能显示的地图格式
                const newMap: Record<string, Message> = {};
                historyData.forEach((msg: any) => {
                    newMap[msg.id] = msg;
                });
                
                setMessagesMap(newMap);
                // 把显示画面定位到最后一条消息
                if (historyData.length > 0) {
                    setCurrentLeafId(historyData[historyData.length - 1].id);
                }
            }
        } catch (e) { 
            console.log("暂时没能从云端拉取到历史"); 
        }
    };
    
    // 只有当配置了 IP 和密码后才去拉取
    if (apiConfig.baseUrl && apiConfig.apiKey) {
        fetchHistory();
    }
  }, [apiConfig.baseUrl, apiConfig.apiKey]);

  // --- 导出 HTML 功能 ---
  const handleExport = () => {
    const currentMessages = getChatChain();
    if (currentMessages.length === 0) {
        alert("空空如也，没什么好存的呀");
        return;
    }
    exportChatToHtml(currentMessages);
    setIsMenuOpen(false);
  };

  // --- 获取对话链的工具函数 ---
  const getChatChain = () => {
    const chain: Message[] = [];
    let currId = currentLeafId;
    while (currId && messagesMap[currId]) {
      chain.unshift(messagesMap[currId]);
      currId = messagesMap[currId].parentId || null;
    }
    return chain;
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;
    
    const userMsgId = Date.now().toString();
    const currentInput = inputValue; 
    setInputValue('');
    setIsMenuOpen(false);
    setIsLoading(true);

    let parentId = currentLeafId; 
    const userMsg: Message = {
      id: userMsgId, 
      role: 'user', 
      content: currentInput, 
      parentId: parentId, 
      childrenIds: [] // <--- 确保这行存在！
    };

  setMessagesMap(prev => {
        const newMap = { ...prev, [userMsgId]: userMsg };
        if (parentId && newMap[parentId]) {
            // --- 核心修正：如果数组不存在，先给它个空数组，防止 push 报错 ---
            if (!newMap[parentId].childrenIds) {
                newMap[parentId].childrenIds = [];
            }
            newMap[parentId].childrenIds.push(userMsgId);
            newMap[parentId].selectedChildId = userMsgId;
        }
        return newMap;
    });
    setCurrentLeafId(userMsgId);
    try {
        const url = `${apiConfig.baseUrl.replace(/\/$/, '')}/chat`;
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-access-token': apiConfig.apiKey 
            },
            body: JSON.stringify({ session_id: "default_user", message: currentInput })
        });

        const data = await response.json();
        const replyText = data.reply;

        const assistantMsgId = (Date.now() + 1).toString();
        const assistantMsg: Message = {
            id: assistantMsgId, role: 'assistant', content: replyText,
            parentId: userMsgId, childrenIds: []
        };

        setMessagesMap(prev => {
            const newMap = { ...prev, [assistantMsgId]: assistantMsg };
            newMap[userMsgId].childrenIds.push(assistantMsgId);
            newMap[userMsgId].selectedChildId = assistantMsgId;
            return newMap;
        });
        setCurrentLeafId(assistantMsgId);
    } catch (error: any) {
        alert("连接断了...");
    } finally {
        setIsLoading(false);
    }
  };

  const handleMenuSelect = (option: string) => {
    setIsMenuOpen(false);
    if (option === 'api') setIsConfigOpen(true);
    if (option === 'system') setIsSystemPromptOpen(true);
    if (option === 'interface') setIsInterfaceConfigOpen(true);
    if (option === 'export') handleExport(); // 这里触发导出
  };

  const currentMessages = useMemo(() => getChatChain(), [messagesMap, currentLeafId]);

  return (
    <div className={`flex flex-col h-screen ${interfaceConfig.pureBlack ? 'bg-black' : 'bg-token-main-surface-primary'}`}>
      <Header centerHeader={interfaceConfig.centerHeader} pureBlack={interfaceConfig.pureBlack} />
      <main className="flex-1 overflow-hidden relative">
        {currentMessages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center p-8">
             <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mb-6 shadow-xl">
                <OpenAIIcon className="w-10 h-10 text-black" />
             </div>
             <h1 className="text-2xl font-semibold text-token-text-primary">凌夜已就绪。</h1>
          </div>
        ) : (
          <MessageList messages={currentMessages} showAvatar={interfaceConfig.showAvatar} messageMap={messagesMap} />
        )}
      </main>
      <div className="pb-safe">
        <div className="px-3 py-2">
            <InputBar 
                value={inputValue} 
                onChange={(e) => setInputValue(e.target.value)} 
                onPlusClick={() => setIsMenuOpen(!isMenuOpen)}
                isLoading={isLoading}
                onSend={handleSend}
                onStop={() => setIsLoading(false)}
                isMenuOpen={isMenuOpen}
                onMenuSelect={handleMenuSelect as any}
                onCloseMenu={() => setIsMenuOpen(false)}
                pureBlack={interfaceConfig.pureBlack}
            />
        </div>
      </div>
      {/* 弹窗部分保持不变 */}
      <ApiConfigModal isOpen={isConfigOpen} onClose={() => setIsConfigOpen(false)} initialConfig={apiConfig} onSave={(c) => {setApiConfig(c); setIsConfigOpen(false);}} />
      <InterfaceConfigModal isOpen={isInterfaceConfigOpen} onClose={() => setIsInterfaceConfigOpen(false)} initialConfig={interfaceConfig} onSave={(c) => {setInterfaceConfig(c); setIsInterfaceConfigOpen(false);}} />
    </div>
  );
};

export default App;