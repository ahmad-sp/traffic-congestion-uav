import React, { useState } from 'react'

const LEVEL_STYLES = {
  GREEN: 'bg-green-500/20 text-green-400 border-green-500/30',
  AMBER: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  EARLY_RED: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  RED: 'bg-red-500/20 text-red-400 border-red-500/30 animate-pulse-red',
}

const BADGE_STYLES = {
  GREEN: 'bg-green-500 text-white',
  AMBER: 'bg-amber-500 text-black',
  EARLY_RED: 'bg-purple-500 text-white',
  RED: 'bg-red-500 text-white',
}

function MetricValue({ label, value }) {
  const hasValue = value !== undefined && value !== null
  return (
    <span className="text-gray-400 text-xs">
      {label}:{' '}
      <span className={hasValue ? 'text-white font-medium' : 'text-gray-600'}>
        {hasValue ? value : '–'}
      </span>
    </span>
  )
}

export default function JunctionGrid({ junctions, armStatus, selectedArm, onSelectArm }) {
  const [expanded, setExpanded] = useState({})

  const toggle = (jid) => {
    setExpanded((prev) => ({ ...prev, [jid]: !prev[jid] }))
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
          Junction Status
        </h2>
        <span className="text-xs text-gray-500">
          Click any arm row to view detailed metrics above
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {junctions.map((junction) => {
          const isExpanded = expanded[junction.junction_id] !== false // default open

          // Junction-level status = highest arm status
          const armLevels = junction.arms.map((a) => {
            const key = `${junction.junction_id}_${a.arm_id}`
            return armStatus[key]?.alert_level || a.alert_level || 'GREEN'
          })
          const junctionLevel = armLevels.includes('RED')
            ? 'RED'
            : armLevels.includes('EARLY_RED')
            ? 'EARLY_RED'
            : armLevels.includes('AMBER')
            ? 'AMBER'
            : 'GREEN'

          return (
            <div
              key={junction.junction_id}
              className={`bg-gray-800 rounded-lg border-2 ${LEVEL_STYLES[junctionLevel]}`}
            >
              {/* Header */}
              <button
                onClick={() => toggle(junction.junction_id)}
                className="w-full px-4 py-3 flex items-center justify-between text-left"
              >
                <div>
                  <span className="font-semibold">{junction.name}</span>
                  <span className="text-gray-500 text-sm ml-2">
                    {junction.junction_id} ({junction.type})
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded text-xs font-bold ${BADGE_STYLES[junctionLevel]}`}>
                    {junctionLevel}
                  </span>
                  <span className="text-gray-500">{isExpanded ? '−' : '+'}</span>
                </div>
              </button>

              {/* Arms */}
              {isExpanded && (
                <div className="border-t border-gray-700 px-4 pb-3">
                  {junction.arms.map((arm) => {
                    const key = `${junction.junction_id}_${arm.arm_id}`
                    const status = armStatus[key]
                    const level = status?.alert_level || arm.alert_level || 'GREEN'
                    const isSelected =
                      selectedArm?.junction_id === junction.junction_id &&
                      selectedArm?.arm_id === arm.arm_id

                    return (
                      <button
                        key={arm.arm_id}
                        onClick={() =>
                          onSelectArm({ junction_id: junction.junction_id, arm_id: arm.arm_id })
                        }
                        className={`w-full mt-2 px-3 py-2 rounded flex items-center justify-between text-sm transition
                          ${isSelected
                            ? 'bg-blue-500/20 border border-blue-500/50'
                            : 'bg-gray-700/50 hover:bg-gray-700 border border-transparent'
                          }`}
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          <span className={`px-1.5 py-0.5 rounded text-xs font-bold flex-shrink-0 ${BADGE_STYLES[level]}`}>
                            {level}
                          </span>
                          <span className="truncate">{arm.name}</span>
                        </div>

                        <div className="flex items-center gap-3 ml-3 flex-shrink-0">
                          <MetricValue label="VPM" value={status?.VPM} />
                          <MetricValue label="Queue" value={status?.queue_depth} />
                          {isSelected && (
                            <span className="text-blue-400 text-xs">▲ viewing</span>
                          )}
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
