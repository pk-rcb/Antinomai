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

  // Poll health on mount
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API}/api/health`)
        const data = await res.json()
        setHealth(data)
      } catch {
        setHealth({ status: 'Cannot reach backend', primary_model: null, vision_model: null })
      }
    }
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
    sessionId.current = Math.random().toString(36).slice(2, 10)
    await fetch(`${API}/api/clear?session_id=${sessionId.current}`, { method: 'DELETE' })
  }, [])

  return { messages, streaming, portfolio, lastRoute, health, sentimentCheck, setSentimentCheck, sendMessage, clearChat }
}
