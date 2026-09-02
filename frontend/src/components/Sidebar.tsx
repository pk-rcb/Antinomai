import './Sidebar.css'

interface Props {
  sentimentCheck: boolean
  onToggleSentiment: (v: boolean) => void
  onClear: () => void
  lastRoute: string | null
  health: { status: string; primary_model?: string | null; vision_model?: string | null }
}

const ROUTE_LABELS: Record<string, string> = {
  debate:      '⚖️  Debate Panel',
  vision:      '📈  Vision Analysis',
  fundamental: '🔬  Fundamental',
  portfolio:   '🗂  Portfolio Engine',
  trivia:      '💡  Trivia / Price',
}

export default function Sidebar({ sentimentCheck, onToggleSentiment, onClear, lastRoute, health }: Props) {
  const isOk = health.status === 'ok'

  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <span className="sidebar-logo">📊</span>
        <div>
          <div className="sidebar-title">Antinomai</div>
          <div className="sidebar-subtitle">Research Platform</div>
        </div>
      </div>

      <div className="sidebar-divider" />

      {/* API Health */}
      <div className="sidebar-section">
        <div className="sidebar-label">Status</div>
        <div className={`sidebar-health ${isOk ? 'health-ok' : 'health-err'}`}>
          <span className="health-dot" />
          {isOk ? 'Connected' : 'Offline'}
        </div>
        {isOk && health.primary_model && (
          <div className="sidebar-model">
            <span>🤖</span> {health.primary_model}
          </div>
        )}
        {!isOk && (
          <div className="sidebar-error">{health.status}</div>
        )}
      </div>

      <div className="sidebar-divider" />

      {/* Controls */}
      <div className="sidebar-section">
        <div className="sidebar-label">Options</div>
        <label className="sidebar-toggle">
          <span>Live Sentiment Check</span>
          <input
            type="checkbox"
            checked={sentimentCheck}
            onChange={e => onToggleSentiment(e.target.checked)}
          />
          <span className="toggle-track">
            <span className="toggle-thumb" />
          </span>
        </label>
        <div className="sidebar-hint">Adds social + analyst sentiment layer (+5–8s)</div>
      </div>

      <div className="sidebar-divider" />

      {/* Last route */}
      {lastRoute && (
        <>
          <div className="sidebar-section">
            <div className="sidebar-label">Last Route</div>
            <div className="sidebar-route">{ROUTE_LABELS[lastRoute] ?? lastRoute}</div>
          </div>
          <div className="sidebar-divider" />
        </>
      )}

      {/* Clear */}
      <div className="sidebar-section">
        <button className="sidebar-clear" onClick={onClear}>
          🗑 Clear Chat
        </button>
      </div>

      <div className="sidebar-spacer" />

      <div className="sidebar-footer">
        LangGraph · Groq · Tavily
      </div>
    </aside>
  )
}

