import React, { useState, useEffect } from 'react'
import useWebSocket from './hooks/useWebSocket'
import RoadMap from './components/RoadMap'
import JunctionGrid from './components/JunctionGrid'
import MetricsStrip from './components/MetricsStrip'
import TimeSeriesChart from './components/TimeSeriesChart'
import AlertLog from './components/AlertLog'
import DroneLog from './components/DroneLog'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const { connected, metrics, alerts: wsAlerts, droneTriggers } = useWebSocket()
  const [junctions, setJunctions] = useState([])
  const [selectedArm, setSelectedArm] = useState(null) // {junction_id, arm_id}
  const [alerts, setAlerts] = useState([])
  const [metricsHistory, setMetricsHistory] = useState([])

  // Fetch junctions on mount
  useEffect(() => {
    fetch(`${API}/junctions`)
      .then((r) => r.json())
      .then(setJunctions)
      .catch(() => {})
  }, [])

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

  // Build arm status from WS metrics
  const armStatus = {}
  for (const [key, data] of Object.entries(metrics)) {
    armStatus[key] = data
  }

  const selectedKey = selectedArm
    ? `${selectedArm.junction_id}_${selectedArm.arm_id}`
    : null
  const liveMetrics = selectedKey ? metrics[selectedKey] : null

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-3 flex items-center justify-between">
        <h1 className="text-xl font-bold tracking-tight">
          Traffic Congestion Detection System
        </h1>
        <div className="flex items-center gap-3 text-sm">
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
          <RoadMap junctions={junctions} armStatus={armStatus} onSelectArm={setSelectedArm} />
        </section>

        {/* Junction Grid */}
        <section>
          <JunctionGrid
            junctions={junctions}
            armStatus={armStatus}
            selectedArm={selectedArm}
            onSelectArm={setSelectedArm}
          />
        </section>

        {/* Metrics strip + time series for selected arm */}
        {selectedArm && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-1">
              <MetricsStrip
                arm={selectedArm}
                liveMetrics={liveMetrics}
              />
            </div>
            <div className="lg:col-span-2">
              <TimeSeriesChart
                arm={selectedArm}
                history={metricsHistory}
                alerts={alerts.filter(
                  (a) =>
                    a.junction_id === selectedArm.junction_id &&
                    a.arm_id === selectedArm.arm_id
                )}
              />
            </div>
          </div>
        )}

        {/* Alert log + drone triggers */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <AlertLog alerts={alerts} />
          <DroneLog triggers={droneTriggers} />
        </div>
      </main>
    </div>
  )
}
