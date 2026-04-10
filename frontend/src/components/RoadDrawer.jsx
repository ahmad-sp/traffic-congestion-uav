import { useMapEvents } from 'react-leaflet'

/**
 * Invisible map component that captures click events for road path drawing.
 * Must be rendered inside <MapContainer>.
 * When active=true, map cursor becomes crosshair and clicks add waypoints.
 */
export default function RoadDrawer({ active, onAddPoint }) {
  useMapEvents({
    click(e) {
      if (active) {
        onAddPoint([e.latlng.lat, e.latlng.lng])
      }
    },
  })
  return null
}
