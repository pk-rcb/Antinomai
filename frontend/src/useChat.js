import { useState, useEffect, useRef, useCallback } from 'react'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export function useChat() {
  const [messages,        setMessages]        = useState([])
  const [streaming,       setStreaming]        = useState(false)
  const [portfolio,       setPortfolio]        = useState(null)
  const [lastRoute,       setLastRoute]        = useState(null)
  const [health,          setHealth]           = useState({ status: 'starting', primary_model: null, vision_model: null })
  const [sentimentCheck,  setSentimentCheck]   = useState(false)
  const sessionId = useRef(Math.random().toString(36).slice(2, 10))

  // Poll health on mount and clear previous session data on refresh
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API}/api/health`, { cache: 'no-store' })
        const data = await res.json()
        setHealth(data)
      } catch {
        setHealth({ status: 'Cannot reach backend', primary_model: null, vision_model: null })
      }
    }
    
    // Wipe previous session if it exists
    const lastSession = localStorage.getItem('finrag_last_session')
    if (lastSession) {
      fetch(`${API}/api/clear?session_id=${lastSession}`, { method: 'DELETE' }).catch(console.error)
    }
    localStorage.setItem('finrag_last_session', sessionId.current)
    
    check()
    const id = setInterval(check, 30_000)
    return () => clearInterval(id)
  }, [])

  const sendMessage = useCallback(async (text, file) => {
    if (streaming) return

    // Upload image first if present
    if (file) {
      const fd = new FormData()
      fd.append('file', file)
      await fetch(`${API}/api/upload?session_id=${sessionId.current}`, { method: 'POST', body: fd })
    }

    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setPortfolio(null)
    setStreaming(true)

    // Add empty assistant bubble
    setMessages(prev => [...prev, { role: 'assistant', content: '' }])

    const res = await fetch(`${API}/api/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        message:                text,
        session_id:             sessionId.current,
        enable_sentiment_check: sentimentCheck,
      }),
    })

    if (!res.body) {
      setStreaming(false)
      return
    }

    const reader  = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer    = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n')
      buffer      = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const evt = JSON.parse(line.slice(6))

          if (evt.type === 'token') {
            setMessages(prev => {
              const msgs  = [...prev]
              const last  = msgs[msgs.length - 1]
              msgs[msgs.length - 1] = { ...last, content: last.content + evt.content }
              return msgs
            })
          }

          if (evt.type === 'done') {
            setLastRoute(evt.route)
            setMessages(prev => {
              const msgs = [...prev]
              msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], route: evt.route }
              return msgs
            })
            if (evt.portfolio) setPortfolio(evt.portfolio)
            setStreaming(false)
          }

          if (evt.type === 'error') {
            setMessages(prev => {
              const msgs = [...prev]
              msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content: `Error: ${evt.content}` }
              return msgs
            })
            setStreaming(false)
          }
        } catch { /* ignore malformed SSE lines */ }
      }
    }

    setStreaming(false)
  }, [streaming, sentimentCheck])

  const clearChat = useCallback(async () => {
    setMessages([])
    setPortfolio(null)
    setLastRoute(null)
    
    const oldSession = sessionId.current
    sessionId.current = Math.random().toString(36).slice(2, 10)
    localStorage.setItem('finrag_last_session', sessionId.current)
    
    await fetch(`${API}/api/clear?session_id=${oldSession}`, { method: 'DELETE' })
  }, [])

  return { messages, streaming, portfolio, lastRoute, health, sentimentCheck, setSentimentCheck, sendMessage, clearChat, sessionId: sessionId.current }
}
