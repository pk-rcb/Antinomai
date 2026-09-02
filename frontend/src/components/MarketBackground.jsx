import { useEffect, useRef } from 'react'
import './MarketBackground.css'

const GLOW = {
  debate:      'rgba(245,158,11,0.07)',
  bull_agent:  'rgba(16,185,129,0.07)',
  bear_agent:  'rgba(239,68,68,0.07)',
  fundamental: 'rgba(59,130,246,0.07)',
  portfolio:   'rgba(139,92,246,0.07)',
  vision:      'rgba(236,72,153,0.07)',
  trivia:      'rgba(59,130,246,0.06)',
}

export default function MarketBackground({ route }) {
  const canvasRef = useRef(null)
  const bgRef     = useRef(null)

  // Update glow color on route change
  useEffect(() => {
    const el = bgRef.current
    if (!el) return
    const color = GLOW[route ?? ''] ?? 'rgba(59,130,246,0.06)'
    el.style.setProperty('--glow-color', color)
  }, [route])

  // Draw a slow-moving price-line on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let frame = 0
    let raf

    const resize = () => {
      canvas.width  = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const draw = () => {
      const { width: W, height: H } = canvas
      ctx.clearRect(0, 0, W, H)

      ctx.beginPath()
      ctx.strokeStyle = 'rgba(59,130,246,0.18)'
      ctx.lineWidth   = 1.5

      for (let x = 0; x <= W; x++) {
        const t  = (x / W) * Math.PI * 6 + frame * 0.004
        const y  = H * 0.5
             + Math.sin(t)       * H * 0.18
             + Math.sin(t * 1.7) * H * 0.08
             + Math.sin(t * 0.4) * H * 0.12

        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.stroke()
      frame++
      raf = requestAnimationFrame(draw)
    }
    draw()

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <div className="market-bg" ref={bgRef}>
      <canvas ref={canvasRef} />
    </div>
  )
}
