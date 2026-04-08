import { useEffect, useRef, useState, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/live'

export default function useWebSocket() {
  const wsRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState(null)
  const [metrics, setMetrics] = useState({})
  const [alerts, setAlerts] = useState([])
  const [droneTriggers, setDroneTriggers] = useState([])
  const reconnectRef = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      if (reconnectRef.current) {
        clearTimeout(reconnectRef.current)
        reconnectRef.current = null
      }
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        setLastMessage(msg)

        const key = `${msg.junction_id}_${msg.arm_id}`

        if (msg.type === 'metrics') {
          setMetrics((prev) => ({ ...prev, [key]: msg.data }))
        } else if (msg.type === 'alert') {
          setAlerts((prev) => [msg.data, ...prev].slice(0, 100))
        } else if (msg.type === 'drone_trigger') {
          setDroneTriggers((prev) => [msg.data, ...prev].slice(0, 50))
        }
      } catch (e) {
        // ignore parse errors
      }
    }

    ws.onclose = () => {
      setConnected(false)
      reconnectRef.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (wsRef.current) wsRef.current.close()
      if (reconnectRef.current) clearTimeout(reconnectRef.current)
    }
  }, [connect])

  return { connected, lastMessage, metrics, alerts, droneTriggers }
}
