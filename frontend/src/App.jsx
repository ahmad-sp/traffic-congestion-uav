import React, { useState, useEffect } from 'react'
import useWebSocket from './hooks/useWebSocket'
import RoadMap from './components/RoadMap'
import JunctionGrid from './components/JunctionGrid'
import MetricsStrip from './components/MetricsStrip'
import TimeSeriesChart from './components/TimeSeriesChart'
import AlertLog from './components/AlertLog'
import DroneLog from './components/DroneLog'
import AdminPanel from './components/AdminPanel'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const { connected, metrics, alerts: wsAlerts, droneTriggers } = useWebSocket()
  const [junctions, setJunctions] = useState([])
  const [selectedArm, setSelectedArm] = useState(null) // {junction_id, arm_id}
  const [alerts, setAlerts] = useState([])
  const [metricsHistory, setMetricsHistory] = useState([])

  // REST-seeded arm status — populated at load and periodically refreshed
  // so the grid shows values even before a WS message arrives for each arm
  const [restArmStatus, setRestArmStatus] = useState({})

  // Admin state
  const [adminMode, setAdminMode] = useState(false)
  const [drawingArm, setDrawingArm] = useState(null)
  const [drawingPath, setDrawingPath] = useState([])

  // Fetch junctions on mount
  useEffect(() => {
    fetch(`${API}/junctions`)
      .then((r) => r.json())
      .then(setJunctions)
      .catch(() => {})
  }, [])

  // Auto-select first arm once junctions load (so MetricsStrip is immediately visible)
  useEffect(() => {
    if (junctions.length > 0 && !selectedArm) {
      const first = junctions[0]
      if (first.arms.length > 0) {
        setSelectedArm({ junction_id: first.junction_id, arm_id: first.arms[0].arm_id })
      }
    }
  }, [junctions]) // eslint-disable-line react-hooks/exhaustive-deps

  // Seed arm status from REST on load and refresh every 10 s while WS is disconnected
  const fetchRestArmStatus = () => {
    if (!junctions.length) return
    Promise.all(
      junctions.map((j) =>
        fetch(`${API}/junction/${j.junction_id}/status`)
          .then((r) => r.json())
          .then((d) => ({ junction_id: j.junction_id, arms: d.arms }))
          .catch(() => null)
      )
    ).then((results) => {
      const next = {}
      for (const result of results) {
        if (!result) continue
        for (const [aid, armData] of Object.entries(result.arms)) {
          const key = `${result.junction_id}_${aid}`
          // Flatten: { ...metrics, alert_level }
          next[key] = { ...armData.metrics, alert_level: armData.alert_level }
        }
      }
      setRestArmStatus(next)
    })
  }

  useEffect(() => {
    fetchRestArmStatus()
    // Poll every 10 s as a fallback when WS hasn't delivered for a given arm yet
    const id = setInterval(fetchRestArmStatus, 10_000)
    return () => clearInterval(id)
  }, [junctions]) // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch alerts on mount
  useEffect(() => {
    fetch(`${API}/alerts?limit=50`)
      .then((r) => r.json())
      .then(setAlerts)
      .catch(() => {})
  }, [])

  // Merge WS alerts
  useEffect(() => {
    if (wsAlerts.length > 0) {
      setAlerts((prev) => {
        const ids = new Set(prev.map((a) => a.alert_id))
        const newOnes = wsAlerts.filter((a) => !ids.has(a.alert_id))
        return [...newOnes, ...prev].slice(0, 100)
      })
    }
  }, [wsAlerts])

  // Fetch time series when arm is selected
  useEffect(() => {
    if (!selectedArm) return
    const { junction_id, arm_id } = selectedArm
    fetch(`${API}/metrics/${junction_id}/${arm_id}?minutes=60`)
      .then((r) => r.json())
      .then((data) => setMetricsHistory(data.reverse()))
      .catch(() => {})
  }, [selectedArm])

  // Merge REST-seeded status with live WS metrics (WS takes priority)
  const armStatus = { ...restArmStatus }
  for (const [key, data] of Object.entries(metrics)) {
    armStatus[key] = data
  }

  const selectedKey = selectedArm
    ? `${selectedArm.junction_id}_${selectedArm.arm_id}`
    : null

  // WS data for selected arm, falling back to REST-seeded values
  const liveMetrics = selectedKey
    ? (metrics[selectedKey] || restArmStatus[selectedKey] || null)
    : null

  // Admin callbacks
  const refreshJunctions = () =>
    fetch(`${API}/junctions`)
      .then((r) => r.json())
      .then(setJunctions)
      .catch(() => {})

  const handleArmMoved = (junction_id, arm_id, lat, lon) => {
    fetch(`${API}/admin/junctions/${junction_id}/arms/${arm_id}/location`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gps_lat: lat, gps_lon: lon }),
    }).then(refreshJunctions).catch(() => {})
  }

  const handleAddDrawPoint = (point) => setDrawingPath((prev) => [...prev, point])

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-3 flex items-center justify-between">
        <h1 className="text-xl font-bold tracking-tight">
          Traffic Congestion Detection System
        </h1>
        <div className="flex items-center gap-3 text-sm">
          <button
            onClick={() => { setAdminMode((v) => !v); setDrawingArm(null); setDrawingPath([]) }}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              adminMode ? 'bg-amber-500 text-gray-900' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {adminMode ? 'Admin ON' : 'Admin'}
          </button>
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              connected ? 'bg-green-400' : 'bg-red-400'
            }`}
          />
          <span className="text-gray-400">
            {connected ? 'Live' : 'Disconnected'}
          </span>
        </div>
      </header>

      <main className="p-4 space-y-4 max-w-[1600px] mx-auto">
        {/* Map */}
        <section>
          <RoadMap
            junctions={junctions}
            armStatus={armStatus}
            onSelectArm={setSelectedArm}
            adminMode={adminMode}
            onArmMoved={handleArmMoved}
            drawingArm={drawingArm}
            drawingPath={drawingPath}
            onAddDrawPoint={handleAddDrawPoint}
          />
        </section>

        {/* Admin Panel */}
        {adminMode && (
          <section>
            <AdminPanel
              junctions={junctions}
              onJunctionUpdate={refreshJunctions}
              drawingArm={drawingArm}
              drawingPath={drawingPath}
              onStartDrawing={(jid, aid) => { setDrawingArm({ junction_id: jid, arm_id: aid }); setDrawingPath([]) }}
              onStopDrawing={() => { setDrawingArm(null); setDrawingPath([]) }}
              onUndoPoint={() => setDrawingPath((prev) => prev.slice(0, -1))}
              onClearPath={() => setDrawingPath([])}
            />
          </section>
        )}

        {/* Metrics strip + time series — always visible, auto-selects first arm */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-1">
            <MetricsStrip
              arm={selectedArm}
              liveMetrics={liveMetrics}
              isLive={selectedKey ? !!metrics[selectedKey] : false}
            />
          </div>
          <div className="lg:col-span-2">
            <TimeSeriesChart
              arm={selectedArm}
              history={metricsHistory}
              alerts={selectedArm ? alerts.filter(
                (a) =>
                  a.junction_id === selectedArm.junction_id &&
                  a.arm_id === selectedArm.arm_id
              ) : []}
            />
          </div>
        </div>

        {/* Junction Grid */}
        <section>
          <JunctionGrid
            junctions={junctions}
            armStatus={armStatus}
            selectedArm={selectedArm}
            onSelectArm={setSelectedArm}
          />
        </section>

        {/* Alert log + drone triggers */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <AlertLog alerts={alerts} />
          <DroneLog triggers={droneTriggers} />
        </div>
      </main>
    </div>
  )
}
