import './Sidebar.css'
import VaultPanel from './VaultPanel'

const ROUTE_LABELS = {
  debate:      'Debate Panel',
  vision:      'Vision Analysis',
  fundamental: 'Fundamental',
  portfolio:   'Portfolio Engine',
  trivia:      'Trivia / Price',
  research:    'Research Vault',
}

const ROUTE_COLORS = {
  debate:      '#f59e0b',
  vision:      '#ec4899',
  fundamental: '#3b82f6',
  portfolio:   '#8b5cf6',
  trivia:      '#10b981',
  research:    '#f97316',
}

export default function Sidebar({
  health,
  lastRoute,
  onClear,
  sentimentCheck,
  onToggleSentiment,
  collapsed,
  onToggleCollapse,
  sessionId,
}) {
  const isOk = health.status === 'ok'

  return (
    <>
      {/* Toggle button always visible */}
      <button
        className={`sidebar-toggle-btn ${collapsed ? 'sidebar-toggle-btn--collapsed' : ''}`}
        onClick={onToggleCollapse}
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          {collapsed ? (
            <polyline points="9 18 15 12 9 6" />
          ) : (
            <polyline points="15 18 9 12 15 6" />
          )}
        </svg>
      </button>

      <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
        {/* Brand */}
        <div className="sidebar-brand">
          <div className="sidebar-logo-wrap">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
          </div>
          <div className="sidebar-brand-text">
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
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }}>
                <rect x="3" y="11" width="18" height="10" rx="2"/>
                <path d="M12 11V7"/><circle cx="12" cy="5" r="2"/>
              </svg>
              {health.primary_model}
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
            <span className="toggle-label-text">Live Sentiment</span>
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
              <div className="sidebar-route" style={{ color: ROUTE_COLORS[lastRoute] ?? 'var(--gold)' }}>
                <span className="route-dot" style={{ background: ROUTE_COLORS[lastRoute] ?? 'var(--gold)' }} />
                {ROUTE_LABELS[lastRoute] ?? lastRoute}
              </div>
            </div>
            <div className="sidebar-divider" />
          </>
        )}

        {/* Clear */}
        <div className="sidebar-section">
          <button className="sidebar-clear" onClick={onClear}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
              <path d="M10 11v6"/><path d="M14 11v6"/>
              <path d="M9 6V4h6v2"/>
            </svg>
            Clear Chat
          </button>
        </div>

        <div className="sidebar-spacer" />

        <div className="sidebar-group-content">
          <VaultPanel sessionId={sessionId} />
        </div>

        <div className="sidebar-footer">
          LangGraph · Groq · Tavily · ChromaDB
        </div>
      </aside>
    </>
  )
}
