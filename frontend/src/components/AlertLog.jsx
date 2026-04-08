import React, { useState } from 'react'

const BADGE = {
  GREEN: 'bg-green-500 text-white',
  AMBER: 'bg-amber-500 text-black',
  EARLY_RED: 'bg-purple-500 text-white',
  RED: 'bg-red-500 text-white',
}

const LEVEL_DISPLAY = {
  GREEN: 'GREEN',
  AMBER: 'AMBER',
  EARLY_RED: 'EARLY RED',
  RED: 'RED',
}

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function AlertLog({ alerts }) {
  const [selected, setSelected] = useState(null)
  const [feedback, setFeedback] = useState({})

  const submitFeedback = async (alertId, confirmed) => {
    try {
      await fetch(`${API}/feedback/${alertId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirmed, notes: '' }),
      })
      setFeedback((prev) => ({ ...prev, [alertId]: confirmed }))
    } catch (e) {
      // ignore
    }
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">Alert Log</h3>

      <div className="overflow-x-auto max-h-80 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="text-gray-400 border-b border-gray-700">
            <tr>
              <th className="text-left py-2 px-2">Time</th>
              <th className="text-left py-2 px-2">Junction</th>
              <th className="text-left py-2 px-2">Arm</th>
              <th className="text-left py-2 px-2">Level</th>
              <th className="text-left py-2 px-2">Type</th>
              <th className="text-left py-2 px-2">Warrants</th>
              <th className="text-left py-2 px-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {(alerts || []).map((alert) => {
              const isRed = alert.level === 'RED'
              const confirmed = feedback[alert.alert_id] ?? alert.confirmed
              const unconfirmedRed = isRed && confirmed === null

              return (
                <tr
                  key={alert.alert_id}
                  onClick={() => setSelected(alert)}
                  className={`border-b border-gray-700/50 cursor-pointer hover:bg-gray-700/30 transition
                    ${unconfirmedRed ? 'animate-pulse' : ''}`}
                >
                  <td className="py-1.5 px-2 text-gray-400">
                    {alert.timestamp?.split('T')[1]?.substring(0, 8) || ''}
                  </td>
                  <td className="py-1.5 px-2">{alert.junction_id}</td>
                  <td className="py-1.5 px-2">{alert.arm_id}</td>
                  <td className="py-1.5 px-2">
                    <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${BADGE[alert.level] || BADGE.AMBER}`}>
                      {LEVEL_DISPLAY[alert.level] || alert.level}
                    </span>
                  </td>
                  <td className="py-1.5 px-2 text-gray-300">{alert.congestion_type || '–'}</td>
                  <td className="py-1.5 px-2 text-gray-400">
                    {(alert.active_warrants || []).join(', ')}
                  </td>
                  <td className="py-1.5 px-2">
                    {confirmed === true && <span className="text-green-400">Confirmed</span>}
                    {confirmed === false && <span className="text-gray-500">Dismissed</span>}
                    {confirmed === null && isRed && (
                      <div className="flex gap-1">
                        <button
                          onClick={(e) => { e.stopPropagation(); submitFeedback(alert.alert_id, true) }}
                          className="px-1.5 py-0.5 bg-green-600 rounded text-white hover:bg-green-500"
                        >
                          Confirm
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); submitFeedback(alert.alert_id, false) }}
                          className="px-1.5 py-0.5 bg-gray-600 rounded text-white hover:bg-gray-500"
                        >
                          Dismiss
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Selected alert detail */}
      {selected && (
        <div className="mt-3 bg-gray-700/50 rounded p-3 text-xs">
          <div className="flex justify-between items-start">
            <div>
              <strong>Alert {selected.alert_id?.substring(0, 8)}</strong>
              <span className="text-gray-400 ml-2">{selected.timestamp}</span>
            </div>
            <button onClick={() => setSelected(null)} className="text-gray-400 hover:text-white">
              Close
            </button>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 text-gray-300">
            <div>LSTM Score: {selected.lstm_score?.toFixed(4)}</div>
            <div>Anomaly Score: {selected.anomaly_score?.toFixed(4)}</div>
            <div>VPM: {selected.current_vpm}</div>
            <div>Queue Depth: {selected.queue_depth}</div>
          </div>
        </div>
      )}
    </div>
  )
}
