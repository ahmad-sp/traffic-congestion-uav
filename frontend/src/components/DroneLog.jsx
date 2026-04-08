import React, { useState } from 'react'

export default function DroneLog({ triggers }) {
  const [expanded, setExpanded] = useState(null)

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">Drone Trigger Log</h3>

      {(!triggers || triggers.length === 0) ? (
        <p className="text-gray-500 text-sm">No drone triggers yet</p>
      ) : (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {triggers.map((t) => (
            <div key={t.trigger_id} className="bg-gray-700/50 rounded p-3 text-xs">
              <div className="flex justify-between items-start">
                <div>
                  <span className="bg-red-500 text-white px-1.5 py-0.5 rounded text-xs font-bold mr-2">
                    DRONE
                  </span>
                  <span className="font-mono">{t.trigger_id?.substring(0, 8)}</span>
                  <span className="text-gray-400 ml-2">
                    {t.junction_id} / {t.arm_id}
                  </span>
                </div>
                <button
                  onClick={() => setExpanded(expanded === t.trigger_id ? null : t.trigger_id)}
                  className="text-blue-400 hover:text-blue-300"
                >
                  {expanded === t.trigger_id ? 'Hide' : 'Details'}
                </button>
              </div>

              <div className="mt-1 text-gray-400">
                Type: <span className="text-gray-200">{t.congestion_type}</span>
                {' | '}
                Severity: <span className="text-gray-200">{t.severity_score?.toFixed(3)}</span>
                {' | '}
                VPM: <span className="text-gray-200">{t.current_VPM}</span>
                {' | '}
                {t.timestamp_iso?.split('T')[1]?.substring(0, 8)}
              </div>

              {expanded === t.trigger_id && (
                <div className="mt-2">
                  <pre className="bg-gray-900 rounded p-2 text-gray-300 overflow-x-auto text-[10px]">
                    {JSON.stringify(t, null, 2)}
                  </pre>
                  {t.evidence_clip_path && (
                    <div className="mt-1">
                      <a
                        href={`/evidence/${t.evidence_clip_path}`}
                        className="text-blue-400 hover:underline"
                      >
                        Evidence clip
                      </a>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
