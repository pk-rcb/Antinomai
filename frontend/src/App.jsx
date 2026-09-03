import { useState } from 'react'
import './index.css'
import './App.css'
import { useChat }        from './useChat'
import MarketBackground   from './components/MarketBackground'
import Sidebar            from './components/Sidebar'
import ChatWindow         from './components/ChatWindow'
import ChatInput          from './components/ChatInput'

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const {
    messages, streaming, portfolio,
    lastRoute, health, sentimentCheck,
    setSentimentCheck, sendMessage, clearChat, sessionId
  } = useChat()

  return (
    <>
      <MarketBackground route={lastRoute} />
      <div className="app-layout">
        <Sidebar
          sentimentCheck={sentimentCheck}
          onToggleSentiment={setSentimentCheck}
          onClear={clearChat}
          lastRoute={lastRoute}
          health={health}
          sessionId={sessionId}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(v => !v)}
        />
        <div className={`app-main ${sidebarCollapsed ? 'app-main--expanded' : ''}`}>
          <ChatWindow
            messages={messages}
            streaming={streaming}
            portfolio={portfolio}
          />
          <ChatInput
            onSend={sendMessage}
            disabled={streaming || health.status !== 'ok'}
          />
        </div>
      </div>
    </>
  )
}
