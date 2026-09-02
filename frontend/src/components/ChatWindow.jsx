import { useRef, useEffect } from 'react'
import MessageBubble from './MessageBubble'
import PortfolioTable from './PortfolioTable'
import './ChatWindow.css'

export default function ChatWindow({ messages, streaming, portfolio }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming])

  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div className="chat-empty">
          <div className="chat-empty-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
          </div>
          <h2>Antinomai</h2>
          <p>Institutional Multi-Agent Research Platform</p>
          <div className="chat-hints">
            <button className="chat-hint-chip" onClick={() => {}}>Ask a stock price</button>
            <button className="chat-hint-chip" onClick={() => {}}>Should I buy TCS?</button>
            <button className="chat-hint-chip" onClick={() => {}}>Analyse AAPL fundamentals</button>
            <button className="chat-hint-chip" onClick={() => {}}>Portfolio: 100 INFY, 50 TCS</button>
          </div>
        </div>
      )}

      {messages.map((msg, i) => {
        const isLast   = i === messages.length - 1
        const isStream = isLast && streaming && msg.role === 'assistant'
        return (
          <MessageBubble
            key={i}
            msg={msg}
            streaming={isStream}
          />
        )
      })}

      {portfolio && (
        <div style={{ maxWidth: 820, margin: '0 auto 20px', padding: '0 8px' }}>
          <PortfolioTable data={portfolio} />
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
