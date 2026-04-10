import React from 'react'

function Gauge({ label, value, max, unit, color }) {
  const hasValue = value !== undefined && value !== null
  const pct = hasValue ? Math.min(100, (value / max) * 100) : 0
  return (
    <div className="bg-gray-700/50 rounded p-3">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="text-2xl font-bold" style={{ color: hasValue ? color : '#4b5563' }}>
        {hasValue
          ? (typeof value === 'number' ? value.toFixed(value < 10 ? 2 : 0) : value)
          : '–'}
        {hasValue && <span className="text-sm text-gray-400 ml-1">{unit}</span>}
      </div>
      <div className="mt-2 h-2 bg-gray-600 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: hasValue ? color : 'transparent' }}
        />
      </div>
    </div>
  )
}

function ScoreBar({ label, value, threshold, maxVal }) {
  const hasValue = value !== undefined && value !== null
  const pct = hasValue ? Math.min(100, (value / maxVal) * 100) : 0
  const threshPct = (threshold / maxVal) * 100
  const isAbove = hasValue && value > threshold
  const color = isAbove ? '#ef4444' : '#22c55e'

  return (
    <div className="bg-gray-700/50 rounded p-3">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="text-2xl font-bold" style={{ color: hasValue ? color : '#4b5563' }}>
        {hasValue ? value.toFixed(4) : '–'}
      </div>
      <div className="mt-2 h-2 bg-gray-600 rounded-full overflow-hidden relative">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: hasValue ? color : 'transparent' }}
        />
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
  const hasValue = value !== undefined && value !== null
  const pct = hasValue ? Math.min(100, (value / 1) * 100) : 0
  const color = !hasValue ? '#4b5563' : value > 0.65 ? '#8B5CF6' : value > 0.4 ? '#f59e0b' : '#22c55e'

  return (
    <div className="bg-gray-700/50 rounded p-3">
      <div className="text-xs text-gray-400 mb-1">Extreme Risk (10 min)</div>
      <div className="text-2xl font-bold" style={{ color }}>
        {hasValue ? value.toFixed(3) : '–'}
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

export default function MetricsStrip({ arm, liveMetrics, isLive }) {
  if (!arm) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 flex items-center justify-center h-full min-h-[200px]">
        <p className="text-gray-500 text-sm text-center">
          Select a camera arm from the map or grid below to view live metrics.
        </p>
      </div>
    )
  }

  // Distinguish null (no data at all) from 0 (genuinely zero traffic)
  const m = liveMetrics || {}
  const hasAnyData = liveMetrics !== null && liveMetrics !== undefined

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-300">
          Live Metrics — {arm.junction_id} / {arm.arm_id}
        </h3>
        <span className={`text-xs px-2 py-0.5 rounded-full ${
          isLive
            ? 'bg-green-500/20 text-green-400'
            : hasAnyData
            ? 'bg-gray-600/50 text-gray-400'
            : 'bg-gray-700 text-gray-500'
        }`}>
          {isLive ? '● Live' : hasAnyData ? 'Last known' : 'No data yet'}
        </span>
      </div>

      <Gauge
        label="Vehicles Per Minute"
        value={hasAnyData ? (m.VPM ?? null) : null}
        max={50}
        unit="VPM"
        color="#3b82f6"
      />

      <Gauge
        label="Queue Depth"
        value={hasAnyData ? (m.queue_depth ?? null) : null}
        max={20}
        unit="vehicles"
        color="#f59e0b"
      />

      <ScoreBar
        label="Congestion Score (LSTM)"
        value={hasAnyData ? (m.lstm_score ?? null) : null}
        threshold={0.7}
        maxVal={1}
      />

      <ScoreBar
        label="Anomaly Score (AE)"
        value={hasAnyData ? (m.anomaly_score ?? null) : null}
        threshold={0.004}
        maxVal={0.02}
      />

      <ExtremeRiskGauge
        value={hasAnyData ? (m.extreme_congestion_risk ?? null) : null}
      />
    </div>
  )
}
