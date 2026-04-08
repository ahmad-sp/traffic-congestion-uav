import React from 'react'
import { MapContainer, TileLayer, Marker, Popup, Polyline, CircleMarker } from 'react-leaflet'

const LEVEL_COLORS = {
  GREEN: '#22c55e',
  AMBER: '#f59e0b',
  EARLY_RED: '#8B5CF6',
  RED: '#ef4444',
}

export default function RoadMap({ junctions, armStatus, onSelectArm }) {
  if (!junctions.length) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-500">
        Loading map...
      </div>
    )
  }

  // Calculate center from all arms
  const allCoords = junctions.flatMap((j) =>
    j.arms.map((a) => [a.gps_lat, a.gps_lon])
  )
  const centerLat = allCoords.reduce((s, c) => s + c[0], 0) / allCoords.length
  const centerLon = allCoords.reduce((s, c) => s + c[1], 0) / allCoords.length

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      <MapContainer center={[centerLat, centerLon]} zoom={15} scrollWheelZoom={true}>
        <TileLayer
          attribution='&copy; OpenStreetMap'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {junctions.map((junction) => {
          // Junction center = average of arm positions
          const jLat = junction.arms.reduce((s, a) => s + a.gps_lat, 0) / junction.arms.length
          const jLon = junction.arms.reduce((s, a) => s + a.gps_lon, 0) / junction.arms.length

          return (
            <React.Fragment key={junction.junction_id}>
              {/* Junction center marker */}
              <CircleMarker
                center={[jLat, jLon]}
                radius={8}
                fillColor="#3b82f6"
                fillOpacity={0.8}
                color="#1e40af"
                weight={2}
              >
                <Popup>
                  <div className="text-sm">
                    <strong>{junction.name}</strong>
                    <br />
                    Type: {junction.type} | ID: {junction.junction_id}
                    <br />
                    Arms: {junction.arms.length}
                  </div>
                </Popup>
              </CircleMarker>

              {/* Arm polylines from junction center to arm position */}
              {junction.arms.map((arm) => {
                const key = `${junction.junction_id}_${arm.arm_id}`
                const status = armStatus[key]
                const level = status?.alert_level || arm.alert_level || 'GREEN'
                const color = LEVEL_COLORS[level]

                return (
                  <React.Fragment key={key}>
                    <Polyline
                      positions={[[jLat, jLon], [arm.gps_lat, arm.gps_lon]]}
                      color={color}
                      weight={5}
                      opacity={0.8}
                      eventHandlers={{
                        click: () => onSelectArm({ junction_id: junction.junction_id, arm_id: arm.arm_id }),
                      }}
                    />
                    <CircleMarker
                      center={[arm.gps_lat, arm.gps_lon]}
                      radius={5}
                      fillColor={color}
                      fillOpacity={1}
                      color="#fff"
                      weight={1}
                      eventHandlers={{
                        click: () => onSelectArm({ junction_id: junction.junction_id, arm_id: arm.arm_id }),
                      }}
                    >
                      <Popup>
                        <div className="text-sm">
                          <strong>{arm.name}</strong>
                          <br />
                          Status: <span style={{ color }}>{level}</span>
                          {status && (
                            <>
                              <br />
                              VPM: {status.VPM} | Queue: {status.queue_depth}
                            </>
                          )}
                        </div>
                      </Popup>
                    </CircleMarker>
                  </React.Fragment>
                )
              })}
            </React.Fragment>
          )
        })}
      </MapContainer>
    </div>
  )
}
