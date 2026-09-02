import { useState, useRef, type KeyboardEvent } from 'react'
import './ChatInput.css'

interface Props {
  onSend:    (text: string, file?: File) => void
  disabled:  boolean
}

export default function ChatInput({ onSend, disabled }: Props) {
  const [text,    setText]   = useState('')
  const [file,    setFile]   = useState<File | null>(null)
  const fileRef              = useRef<HTMLInputElement>(null)
  const [isRecording, setIsRecording] = useState(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const submit = () => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed, file ?? undefined)
    setText('')
    setFile(null)
    if (fileRef.current) fileRef.current.value = ''
  }

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
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
          // Adjust backend URL if hosted elsewhere; using relative/local for now
          const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
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
             console.error("Transcription error", await res.text())
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
          <span>📎 {file.name}</span>
          <button onClick={() => { setFile(null); if (fileRef.current) fileRef.current.value = '' }}>✕</button>
        </div>
      )}
      <div className="chat-input-bar">
        <button
          className="chat-btn-icon"
          title="Upload chart"
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
        >
          📎
        </button>
        <button
          className={`chat-btn-icon ${isRecording ? 'recording' : ''}`}
          title={isRecording ? 'Stop recording' : 'Record voice'}
          onClick={toggleRecording}
          disabled={disabled && !isRecording}
          style={{ color: isRecording ? '#f87171' : 'inherit' }}
        >
          🎙️
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
          placeholder="Ask about a stock, portfolio, or upload a chart…"
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={onKey}
          rows={1}
          disabled={disabled}
        />
        <button
          className={`chat-btn-send ${disabled ? '' : 'chat-btn-send--active'}`}
          onClick={submit}
          disabled={disabled || !text.trim()}
        >
          {disabled ? '⏳' : '↑'}
        </button>
      </div>
      <p className="chat-input-hint">Enter to send · Shift+Enter for new line</p>
    </div>
  )
}
