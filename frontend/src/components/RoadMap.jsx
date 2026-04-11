import React, { useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, Polyline, CircleMarker } from 'react-leaflet'
import L from 'leaflet'
import RoadDrawer from './RoadDrawer'

const LEVEL_COLORS = {
  GREEN: '#22c55e',
  AMBER: '#f59e0b',
  EARLY_RED: '#8B5CF6',
  RED: '#ef4444',
}

function makeDivIcon(color) {
  return L.divIcon({
    className: '',
    html: `<div style="width:14px;height:14px;border-radius:50%;background:${color};border:2px solid #fff;cursor:grab;box-shadow:0 0 4px rgba(0,0,0,0.5);"></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  })
}

export default function RoadMap({
  junctions,
  armStatus,
  onSelectArm,
  adminMode = false,
  onArmMoved,
  drawingArm,
  drawingPath = [],
  onAddDrawPoint,
}) {
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

  const isDrawingActive = adminMode && drawingArm !== null

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      <style>{`
        .leaflet-container { height: 420px; }
        ${isDrawingActive ? '.leaflet-container { cursor: crosshair !important; }' : ''}
      `}</style>
      <MapContainer center={[centerLat, centerLon]} zoom={15} scrollWheelZoom={true}>
        <TileLayer
          attribution='&copy; OpenStreetMap'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Road drawing click handler */}
        <RoadDrawer active={isDrawingActive} onAddPoint={onAddDrawPoint} />

        {/* In-progress drawing overlay */}
        {drawingPath.map(([lat, lon], i) => (
          <CircleMarker
            key={`draw-pt-${i}`}
            center={[lat, lon]}
            radius={4}
            fillColor="#fff"
            fillOpacity={1}
            color="#f59e0b"
            weight={2}
          />
        ))}
        {drawingPath.length >= 2 && (
          <Polyline
            positions={drawingPath}
            color="#f59e0b"
            weight={4}
            dashArray="8 6"
            opacity={0.9}
          />
        )}

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

              {/* Arm road lines and position markers */}
              {junction.arms.map((arm) => {
                const key = `${junction.junction_id}_${arm.arm_id}`
                const status = armStatus[key]
                const level = status?.alert_level || arm.alert_level || 'GREEN'
                const color = LEVEL_COLORS[level]
                const hasRoadPath = arm.road_path && arm.road_path.length >= 2

                return (
                  <React.Fragment key={key}>
                    {/* Road line — thick traffic layer if road_path defined, else fallback thin line */}
                    {hasRoadPath ? (
                      <Polyline
                        key={`${key}_road_${color}`}
                        positions={arm.road_path}
                        pathOptions={{ color, weight: 8, opacity: 0.85, lineCap: 'round', lineJoin: 'round' }}
                        eventHandlers={{
                          click: () => onSelectArm({ junction_id: junction.junction_id, arm_id: arm.arm_id }),
                        }}
                      />
                    ) : (
                      <Polyline
                        key={`${key}_line_${color}`}
                        positions={[[jLat, jLon], [arm.gps_lat, arm.gps_lon]]}
                        pathOptions={{ color, weight: 5, opacity: 0.8 }}
                        eventHandlers={{
                          click: () => onSelectArm({ junction_id: junction.junction_id, arm_id: arm.arm_id }),
                        }}
                      />
                    )}

                    {/* Arm endpoint marker — draggable in admin mode */}
                    {adminMode ? (
                      <Marker
                        position={[arm.gps_lat, arm.gps_lon]}
                        icon={makeDivIcon(color)}
                        draggable={true}
                        eventHandlers={{
                          dragend(e) {
                            const { lat, lng } = e.target.getLatLng()
                            if (onArmMoved) {
                              onArmMoved(junction.junction_id, arm.arm_id, lat, lng)
                            }
                          },
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
                            <br />
                            <em className="text-gray-500">Drag to reposition</em>
                          </div>
                        </Popup>
                      </Marker>
                    ) : (
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
                    )}
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
