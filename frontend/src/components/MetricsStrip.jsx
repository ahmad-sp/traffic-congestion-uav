import React from 'react'

function Gauge({ label, value, max, unit, color }) {
  const pct = Math.min(100, (value / max) * 100)
  return (
    <div className="bg-gray-700/50 rounded p-3">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="text-2xl font-bold" style={{ color }}>
        {typeof value === 'number' ? value.toFixed(value < 10 ? 2 : 0) : '–'}
        <span className="text-sm text-gray-400 ml-1">{unit}</span>
      </div>
      <div className="mt-2 h-2 bg-gray-600 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

function ScoreBar({ label, value, threshold, maxVal }) {
  const pct = Math.min(100, (value / maxVal) * 100)
  const threshPct = (threshold / maxVal) * 100
  const isAbove = value > threshold
  const color = isAbove ? '#ef4444' : '#22c55e'

  return (
    <div className="bg-gray-700/50 rounded p-3">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="text-2xl font-bold" style={{ color }}>
        {typeof value === 'number' ? value.toFixed(4) : '–'}
      </div>
      <div className="mt-2 h-2 bg-gray-600 rounded-full overflow-hidden relative">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
        {/* Threshold line */}
        <div
          className="absolute top-0 h-full w-0.5 bg-yellow-400"
          style={{ left: `${threshPct}%` }}
          title={`Threshold: ${threshold}`}
        />
      </div>
    </div>
  )
}

function ExtremeRiskGauge({ value }) {
  const pct = Math.min(100, (value / 1) * 100)
  const color = value > 0.65 ? '#8B5CF6' : value > 0.4 ? '#f59e0b' : '#22c55e'

  return (
    <div className="bg-gray-700/50 rounded p-3">
      <div className="text-xs text-gray-400 mb-1">Extreme Risk (10 min)</div>
      <div className="text-2xl font-bold" style={{ color }}>
        {typeof value === 'number' ? value.toFixed(3) : '–'}
      </div>
      <div className="mt-2 h-2 bg-gray-600 rounded-full overflow-hidden relative">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
        <div
          className="absolute top-0 h-full w-0.5 bg-purple-400"
          style={{ left: '65%' }}
          title="Threshold: 0.65"
        />
      </div>
    </div>
  )
}

export default function MetricsStrip({ arm, liveMetrics }) {
  const m = liveMetrics || {}

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3">
      <h3 className="text-sm font-semibold text-gray-300 mb-2">
        Live Metrics — {arm.junction_id} / {arm.arm_id}
      </h3>

      <Gauge
        label="Vehicles Per Minute"
        value={m.VPM ?? 0}
        max={50}
        unit="VPM"
        color="#3b82f6"
      />

      <Gauge
        label="Queue Depth"
        value={m.queue_depth ?? 0}
        max={20}
        unit="vehicles"
        color="#f59e0b"
      />

      <ScoreBar
        label="Congestion Score (LSTM)"
        value={m.lstm_score ?? 0}
        threshold={0.7}
        maxVal={1}
      />

      <ScoreBar
        label="Anomaly Score (AE)"
        value={m.anomaly_score ?? 0}
        threshold={0.004}
        maxVal={0.02}
      />

      <ExtremeRiskGauge
        value={m.extreme_congestion_risk ?? 0}
      />
    </div>
  )
}
