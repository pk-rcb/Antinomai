import ReactMarkdown from 'react-markdown'
import remarkGfm    from 'remark-gfm'
import './MessageBubble.css'

const ROUTE_LABELS = {
  debate:      { label: 'DEBATE',      color: '#f59e0b' },
  vision:      { label: 'VISION',      color: '#ec4899' },
  fundamental: { label: 'FUNDAMENTAL', color: '#3b82f6' },
  portfolio:   { label: 'PORTFOLIO',   color: '#8b5cf6' },
  trivia:      { label: 'TRIVIA',      color: '#10b981' },
}

export default function MessageBubble({ msg, streaming }) {
  const isUser = msg.role === 'user'
  const tag    = msg.route ? ROUTE_LABELS[msg.route] : null

  return (
    <div className={`bubble-wrap ${isUser ? 'bubble-user' : 'bubble-ai'}`}>
      <div className="bubble-avatar">
        {isUser ? (
          /* Person icon */
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
          </svg>
        ) : (
          /* Bot icon */
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="11" width="18" height="10" rx="2"/>
            <path d="M12 11V7"/>
            <circle cx="12" cy="5" r="2"/>
            <line x1="8" y1="15" x2="8" y2="17"/>
            <line x1="16" y1="15" x2="16" y2="17"/>
          </svg>
        )}
      </div>
      <div className="bubble-body">
        {tag && !isUser && (
          <span className="bubble-tag" style={{ borderColor: tag.color, color: tag.color }}>
            {tag.label}
          </span>
        )}
        <div className={`bubble-content ${isUser ? 'bubble-content--user' : 'bubble-content--ai'}`}>
          {isUser ? (
            <>
              {msg.image && <img src={msg.image} alt="uploaded" className="bubble-image" style={{ maxWidth: '100%', borderRadius: 8, marginBottom: 8 }} />}
              {msg.content && <p>{msg.content}</p>}
            </>
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
