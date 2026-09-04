import { useState, useRef } from 'react'
import './ChatInput.css'

export default function ChatInput({ onSend, disabled }) {
  const [text,        setText]        = useState('')
  const [file,        setFile]        = useState(null)
  const [isRecording, setIsRecording] = useState(false)
  const fileRef          = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef   = useRef([])

  const submit = () => {
    const trimmed = text.trim()
    if ((!trimmed && !file) || disabled) return
    onSend(trimmed, file ?? undefined)
    setText('')
    setFile(null)
    if (fileRef.current) fileRef.current.value = ''
  }

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const onFileChange = (e) => {
    setFile(e.target.files?.[0] ?? null)
  }

  const toggleRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop()
      setIsRecording(false)
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        stream.getTracks().forEach(track => track.stop())

        const formData = new FormData()
        formData.append('file', audioBlob, 'audio.webm')
        try {
          const backendUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
          const res = await fetch(`${backendUrl}/api/transcribe`, {
            method: 'POST',
            body: formData
          })
          if (res.ok) {
            const data = await res.json()
            if (data.text) {
              setText(prev => (prev ? prev + ' ' : '') + data.text)
            }
          } else {
            console.error('Transcription error', await res.text())
          }
        } catch (e) {
          console.error('Transcription failed:', e)
        }
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (err) {
      console.error('Microphone access denied:', err)
      alert('Microphone access denied. Please check permissions.')
    }
  }

  return (
    <div className="chat-input-wrap">
      {file && (
        <div className="chat-input-preview">
          <span className="preview-icon">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/>
            </svg>
          </span>
          <span className="preview-name">{file.name}</span>
          <button className="preview-remove" onClick={() => { setFile(null); if (fileRef.current) fileRef.current.value = '' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
      )}
      <div className="chat-input-bar">
        {/* Attach button */}
        <button
          className="chat-btn-icon"
          title="Attach image"
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/>
          </svg>
        </button>

        {/* Mic button */}
        <button
          className={`chat-btn-icon mic-btn ${isRecording ? 'mic-btn--recording' : ''}`}
          title={isRecording ? 'Stop recording' : 'Voice input'}
          onClick={toggleRecording}
          disabled={disabled && !isRecording}
        >
          {isRecording ? (
            /* Animated waveform bars when recording */
            <span className="mic-recording-icon">
              <span className="mic-bar" />
              <span className="mic-bar" />
              <span className="mic-bar" />
              <span className="mic-bar" />
              <span className="mic-bar" />
            </span>
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="23"/>
              <line x1="8"  y1="23" x2="16" y2="23"/>
            </svg>
          )}
        </button>

        <input
          ref={fileRef}
          type="file"
          accept=".jpg,.jpeg,.png"
          onChange={onFileChange}
          style={{ display: 'none' }}
        />

        <textarea
          className="chat-textarea"
          placeholder="Ask about a stock, portfolio, or upload a chart..."
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={onKey}
          rows={1}
          disabled={disabled}
        />

        {/* Send button */}
        <button
          className={`chat-btn-send ${disabled ? '' : 'chat-btn-send--active'}`}
          onClick={submit}
          disabled={disabled || (!text.trim() && !file)}
        >
          {disabled ? (
            <svg className="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M21 12a9 9 0 11-6.219-8.56"/>
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="19" x2="12" y2="5"/>
              <polyline points="5 12 12 5 19 12"/>
            </svg>
          )}
        </button>
      </div>
      <p className="chat-input-hint">Enter to send · Shift+Enter for new line</p>
    </div>
  )
}
