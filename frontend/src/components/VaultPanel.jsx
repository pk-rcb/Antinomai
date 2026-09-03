import { useState, useEffect, useRef, useCallback } from 'react'
import './VaultPanel.css'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const DOC_TYPE_LABELS = {
  earnings_call:  'Earnings Call',
  annual_report:  'Annual Report',
  research_note:  'Research Note',
  sec_filing:     'SEC Filing',
  other:          'Other',
}

const DOC_TYPE_ICONS = {
  earnings_call:  '🎙️',
  annual_report:  '📊',
  research_note:  '📝',
  sec_filing:     '🏛️',
  other:          '📄',
}

export default function VaultPanel({ sessionId }) {
  const [open,        setOpen]        = useState(false)
  const [docs,        setDocs]        = useState([])
  const [loading,     setLoading]     = useState(false)
  const [uploading,   setUploading]   = useState(false)
  const [deleting,    setDeleting]    = useState(null)   // doc_id being deleted
  const [dragOver,    setDragOver]    = useState(false)
  const [mode,        setMode]        = useState('file') // 'file' | 'text'
  const [ticker,      setTicker]      = useState('')
  const [docType,     setDocType]     = useState('other')
  const [pasteText,   setPasteText]   = useState('')
  const [sourceName,  setSourceName]  = useState('')
  const [feedback,    setFeedback]    = useState(null)   // { type: 'ok'|'err', msg }
  const fileInputRef = useRef(null)

  // Fetch doc list
  const fetchDocs = useCallback(async () => {
    setLoading(true)
    try {
      const qs = sessionId ? `?session_id=${sessionId}` : ''
      const res  = await fetch(`${API}/api/vault/docs${qs}`)
      const data = await res.json()
      setDocs(data.docs ?? [])
    } catch {
      setDocs([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open) fetchDocs()
  }, [open, fetchDocs])

  // Show feedback toast then clear
  const showFeedback = (type, msg) => {
    setFeedback({ type, msg })
    setTimeout(() => setFeedback(null), 3500)
  }

  // Upload a File object
  const uploadFile = async (file) => {
    if (!file) return
    const allowed = ['application/pdf', 'text/plain', 'text/markdown']
    const allowedExt = ['.pdf', '.txt', '.md']
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!allowed.includes(file.type) && !allowedExt.includes(ext)) {
      showFeedback('err', `Unsupported file type: ${ext}. Use PDF, TXT, or MD.`)
      return
    }

    setUploading(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const params = new URLSearchParams()
      if (ticker.trim()) params.set('ticker', ticker.trim().toUpperCase())
      params.set('doc_type', docType)
      if (sessionId) params.set('session_id', sessionId)

      const res  = await fetch(`${API}/api/vault/ingest?${params}`, { method: 'POST', body: fd })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail ?? 'Upload failed')
      showFeedback('ok', `✅ Ingested: ${data.filename}`)
      setTicker('')
      await fetchDocs()
    } catch (e) {
      showFeedback('err', `❌ ${e.message}`)
    } finally {
      setUploading(false)
    }
  }

  // Ingest pasted text
  const uploadText = async () => {
    if (!pasteText.trim() || !sourceName.trim()) {
      showFeedback('err', 'Source name and text content are required.')
      return
    }
    setUploading(true)
    try {
      const params = new URLSearchParams()
      if (ticker.trim()) params.set('ticker', ticker.trim().toUpperCase())
      params.set('doc_type', docType)
      if (sessionId) params.set('session_id', sessionId)

      const res  = await fetch(`${API}/api/vault/ingest-text?${params}`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          text:        pasteText,
          source_name: sourceName,
          ticker:      ticker.trim().toUpperCase() || null,
          doc_type:    docType,
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail ?? 'Ingest failed')
      showFeedback('ok', `✅ Ingested: ${sourceName}`)
      setPasteText('')
      setSourceName('')
      setTicker('')
      await fetchDocs()
    } catch (e) {
      showFeedback('err', `❌ ${e.message}`)
    } finally {
      setUploading(false)
    }
  }

  // Delete a document
  const deleteDoc = async (doc_id, source) => {
    if (!window.confirm(`Remove "${source}" from vault?`)) return
    setDeleting(doc_id)
    try {
      const res = await fetch(`${API}/api/vault/docs/${doc_id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('Delete failed')
      showFeedback('ok', `🗑️ Removed: ${source}`)
      await fetchDocs()
    } catch (e) {
      showFeedback('err', `❌ ${e.message}`)
    } finally {
      setDeleting(null)
    }
  }

  // Drag-and-drop handlers
  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) uploadFile(file)
  }

  return (
    <div className="vault-panel">
      {/* Header toggle */}
      <button
        className={`vault-header ${open ? 'vault-header--open' : ''}`}
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
        id="vault-panel-toggle"
      >
        <div className="vault-header-left">
          <span className="vault-icon">📂</span>
          <span className="vault-title">Research Vault</span>
          {docs.length > 0 && (
            <span className="vault-badge" title={`${docs.length} document(s) indexed`}>
              {docs.length}
            </span>
          )}
        </div>
        <svg
          className={`vault-chevron ${open ? 'vault-chevron--open' : ''}`}
          width="14" height="14" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" strokeWidth="2.5"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {/* Collapsible body */}
      {open && (
        <div className="vault-body">

          {/* Feedback toast */}
          {feedback && (
            <div className={`vault-toast vault-toast--${feedback.type}`}>
              {feedback.msg}
            </div>
          )}

          {/* Mode tabs */}
          <div className="vault-tabs">
            <button
              className={`vault-tab ${mode === 'file' ? 'vault-tab--active' : ''}`}
              onClick={() => setMode('file')}
              id="vault-tab-file"
            >
              Upload File
            </button>
            <button
              className={`vault-tab ${mode === 'text' ? 'vault-tab--active' : ''}`}
              onClick={() => setMode('text')}
              id="vault-tab-text"
            >
              Paste Text
            </button>
          </div>

          {/* Shared metadata fields */}
          <div className="vault-meta-row">
            <input
              className="vault-input vault-input--ticker"
              placeholder="Ticker (opt.)"
              value={ticker}
              onChange={e => setTicker(e.target.value)}
              maxLength={12}
              id="vault-ticker-input"
            />
            <select
              className="vault-input vault-select"
              value={docType}
              onChange={e => setDocType(e.target.value)}
              id="vault-doctype-select"
            >
              {Object.entries(DOC_TYPE_LABELS).map(([val, label]) => (
                <option key={val} value={val}>{label}</option>
              ))}
            </select>
          </div>

          {/* File upload mode */}
          {mode === 'file' && (
            <>
              <div
                className={`vault-dropzone ${dragOver ? 'vault-dropzone--over' : ''} ${uploading ? 'vault-dropzone--uploading' : ''}`}
                onDragOver={e => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => !uploading && fileInputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={e => e.key === 'Enter' && fileInputRef.current?.click()}
                id="vault-dropzone"
              >
                {uploading ? (
                  <div className="vault-uploading">
                    <span className="vault-spinner" />
                    <span>Ingesting…</span>
                  </div>
                ) : (
                  <>
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                      <polyline points="17 8 12 3 7 8"/>
                      <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                    <span className="vault-drop-label">Drop PDF / TXT / MD here</span>
                    <span className="vault-drop-sub">or click to browse</span>
                  </>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.md,application/pdf,text/plain"
                style={{ display: 'none' }}
                onChange={e => { uploadFile(e.target.files[0]); e.target.value = '' }}
                id="vault-file-input"
              />
            </>
          )}

          {/* Paste text mode */}
          {mode === 'text' && (
            <div className="vault-text-mode">
              <input
                className="vault-input"
                placeholder="Source name (e.g. TCS_Q3_2025_Transcript)"
                value={sourceName}
                onChange={e => setSourceName(e.target.value)}
                id="vault-source-name-input"
              />
              <textarea
                className="vault-textarea"
                placeholder="Paste earnings call transcript, research note, or any text…"
                value={pasteText}
                onChange={e => setPasteText(e.target.value)}
                rows={5}
                id="vault-paste-textarea"
              />
              <button
                className="vault-ingest-btn"
                onClick={uploadText}
                disabled={uploading || !pasteText.trim() || !sourceName.trim()}
                id="vault-ingest-text-btn"
              >
                {uploading ? <span className="vault-spinner" /> : null}
                {uploading ? 'Ingesting…' : 'Ingest Text'}
              </button>
            </div>
          )}

          {/* Document list */}
          <div className="vault-docs-section">
            <div className="vault-docs-label">
              {loading ? 'Loading…' : docs.length === 0 ? 'No documents yet' : `${docs.length} document${docs.length !== 1 ? 's' : ''} indexed`}
            </div>

            {loading && <div className="vault-docs-loading"><span className="vault-spinner" /></div>}

            {!loading && docs.length === 0 && (
              <div className="vault-empty">
                <div className="vault-empty-icon">🗂️</div>
                <div className="vault-empty-text">
                  Upload earnings transcripts, annual reports, or research notes
                  to enable the <strong>Research Vault</strong> route in chat.
                </div>
              </div>
            )}

            {!loading && docs.map(doc => (
              <div key={doc.doc_id} className="vault-doc-row">
                <div className="vault-doc-icon">
                  {DOC_TYPE_ICONS[doc.doc_type] ?? '📄'}
                </div>
                <div className="vault-doc-info">
                  <div className="vault-doc-name" title={doc.source}>
                    {doc.source}
                  </div>
                  <div className="vault-doc-meta">
                    {doc.ticker && <span className="vault-doc-ticker">{doc.ticker}</span>}
                    <span className="vault-doc-type">{DOC_TYPE_LABELS[doc.doc_type] ?? doc.doc_type}</span>
                    <span className="vault-doc-date">{doc.date_added}</span>
                    <span className="vault-doc-chunks">{doc.chunk_count} chunks</span>
                  </div>
                </div>
                <button
                  className="vault-doc-delete"
                  onClick={() => deleteDoc(doc.doc_id, doc.source)}
                  disabled={deleting === doc.doc_id}
                  title="Remove document"
                  id={`vault-delete-${doc.doc_id}`}
                >
                  {deleting === doc.doc_id
                    ? <span className="vault-spinner vault-spinner--sm" />
                    : '🗑'
                  }
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
