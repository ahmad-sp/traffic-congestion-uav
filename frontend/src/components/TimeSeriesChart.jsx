import React from 'react'
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'

export default function TimeSeriesChart({ arm, history, alerts }) {
  if (!history.length) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 text-center text-gray-500">
        {arm ? 'Loading historical data…' : 'Select an arm to view time series'}
      </div>
    )
  }

  // Format data for Recharts
  const data = history.map((row) => ({
    time: row.timestamp?.split('T')[1]?.substring(0, 5) || '',
    VPM: row.VPM,
    congestion_score: row.lstm_score ?? 0,
    extreme_risk: row.extreme_congestion_risk ?? 0,
    queue_depth: row.queue_depth,
    occupancy: row.occupancy_pct,
    baseline: row.baseline_85th ?? null,
  }))

  // Alert timestamps for reference lines
  const alertTimes = (alerts || []).map((a) => ({
    time: a.timestamp?.split('T')[1]?.substring(0, 5),
    level: a.level,
  }))

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">
        Time Series — {arm.junction_id} / {arm.arm_id} (last 60 min)
      </h3>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="time" stroke="#9ca3af" tick={{ fontSize: 11 }} />
          <YAxis yAxisId="left" stroke="#9ca3af" tick={{ fontSize: 11 }} />
          <YAxis yAxisId="right" orientation="right" domain={[0, 1]} stroke="#9ca3af" tick={{ fontSize: 11 }} />

          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
            labelStyle={{ color: '#9ca3af' }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />

          {/* VPM line */}
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="VPM"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            name="VPM"
          />

          {/* Baseline dashed line */}
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="baseline"
            stroke="#9ca3af"
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
            name="85th pct baseline"
          />

          {/* Congestion score area */}
          <Area
            yAxisId="right"
            type="monotone"
            dataKey="congestion_score"
            fill="#ef4444"
            fillOpacity={0.15}
            stroke="#ef4444"
            strokeWidth={1}
            name="Congestion Score"
          />

          {/* Extreme risk dashed line */}
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="extreme_risk"
            stroke="#8B5CF6"
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
            name="Extreme Risk"
          />

          {/* Alert reference lines */}
          {alertTimes.map((a, i) => (
            <ReferenceLine
              key={i}
              yAxisId="left"
              x={a.time}
              stroke={a.level === 'RED' ? '#ef4444' : a.level === 'EARLY_RED' ? '#8B5CF6' : '#f59e0b'}
              strokeDasharray="3 3"
              strokeWidth={2}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
