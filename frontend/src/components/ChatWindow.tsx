import { useRef, useEffect } from 'react'
import MessageBubble, { type Message } from './MessageBubble'
import PortfolioTable                from './PortfolioTable'
import './ChatWindow.css'

interface Props {
  messages:   Message[]
  streaming:  boolean
  portfolio:  PortfolioData | null
}

export interface PortfolioData {
  rows:  PortfolioRow[]
  total: number
  beta:  number | null
}

export interface PortfolioRow {
  Ticker:     string
  Shares:     number
  Price:      string
  Value_USD:  number
  Weight_Pct: number
  Beta:       number | null
  Vol_Ann:    number | null
}

export default function ChatWindow({ messages, streaming, portfolio }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming])

  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div className="chat-empty">
          <div className="chat-empty-icon">📊</div>
          <h2>Antinomai</h2>
          <p>Institutional Multi-Agent Research Platform</p>
          <div className="chat-hints">
            <span>💡 Ask a stock price</span>
            <span>⚖️ Should I buy TCS?</span>
            <span>🔬 Analyse AAPL fundamentals</span>
            <span>🗂 Portfolio: 100 INFY, 50 TCS</span>
          </div>
        </div>
      )}

      {messages.map((msg, i) => {
        const isLast    = i === messages.length - 1
        const isStream  = isLast && streaming && msg.role === 'assistant'
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
