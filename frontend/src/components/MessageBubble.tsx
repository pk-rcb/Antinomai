import ReactMarkdown from 'react-markdown'
import remarkGfm    from 'remark-gfm'
import './MessageBubble.css'

export interface Message {
  role:    'user' | 'assistant'
  content: string
  route?:  string
}

const ROUTE_LABELS: Record<string, { label: string; color: string; emoji: string }> = {
  debate:      { label: 'DEBATE',      color: '#f59e0b', emoji: '⚖️' },
  vision:      { label: 'VISION',      color: '#ec4899', emoji: '📈' },
  fundamental: { label: 'FUNDAMENTAL', color: '#3b82f6', emoji: '🔬' },
  portfolio:   { label: 'PORTFOLIO',   color: '#8b5cf6', emoji: '🗂' },
  trivia:      { label: 'TRIVIA',      color: '#10b981', emoji: '💡' },
}

interface Props {
  msg:       Message
  streaming?: boolean
}

export default function MessageBubble({ msg, streaming }: Props) {
  const isUser = msg.role === 'user'
  const tag    = msg.route ? ROUTE_LABELS[msg.route] : null

  return (
    <div className={`bubble-wrap ${isUser ? 'bubble-user' : 'bubble-ai'}`}>
      <div className="bubble-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="bubble-body">
        {tag && !isUser && (
          <span className="bubble-tag" style={{ borderColor: tag.color, color: tag.color }}>
            {tag.emoji} {tag.label}
          </span>
        )}
        <div className={`bubble-content ${isUser ? 'bubble-content--user' : 'bubble-content--ai'}`}>
          {isUser ? (
            <p>{msg.content}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {msg.content}
            </ReactMarkdown>
          )}
          {streaming && <span className="bubble-cursor" />}
        </div>
      </div>
    </div>
  )
}

