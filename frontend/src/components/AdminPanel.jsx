import React, { useState } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const HOURS = Array.from({ length: 24 }, (_, i) => i)

function formatHour(h) {
  return `${String(h).padStart(2, '0')}:00`
}

// ─── Shared sub-components ───

function PeriodChips({ periods, onRemove }) {
  if (!periods || periods.length === 0) {
    return <span className="text-gray-500 text-xs">No periods defined</span>
  }
  return (
    <div className="flex flex-wrap gap-2">
      {periods.map((p, i) => (
        <span
          key={i}
          className="flex items-center gap-1 bg-gray-700 text-gray-200 text-xs px-2 py-1 rounded"
        >
          {formatHour(p[0])}–{formatHour(p[1])}
          <button
            onClick={() => onRemove(i)}
            className="text-gray-400 hover:text-red-400 ml-1"
            title="Remove"
          >
            ✕
          </button>
        </span>
      ))}
    </div>
  )
}

function PeriodAdder({ onAdd }) {
  const [start, setStart] = useState(7)
  const [end, setEnd] = useState(9)

  return (
    <div className="flex items-center gap-2 mt-2">
      <select
        value={start}
        onChange={(e) => setStart(Number(e.target.value))}
        className="bg-gray-700 text-gray-200 text-xs rounded px-2 py-1"
      >
        {HOURS.map((h) => (
          <option key={h} value={h}>{formatHour(h)}</option>
        ))}
      </select>
      <span className="text-gray-400 text-xs">to</span>
      <select
        value={end}
        onChange={(e) => setEnd(Number(e.target.value))}
        className="bg-gray-700 text-gray-200 text-xs rounded px-2 py-1"
      >
        {HOURS.filter((h) => h > start).map((h) => (
          <option key={h} value={h}>{formatHour(h)}</option>
        ))}
      </select>
      <button
        onClick={() => onAdd([start, end])}
        className="bg-blue-600 hover:bg-blue-500 text-white text-xs px-3 py-1 rounded"
      >
        Add
      </button>
    </div>
  )
}

function InfoBanner({ text }) {
  return (
    <div className="bg-amber-900/40 border border-amber-700 text-amber-300 text-xs rounded p-2 mt-2">
      {text}
    </div>
  )
}

// ─── Tab 1: Peak Hours ───

function PeakHoursTab({ junctions, onJunctionUpdate }) {
  const [selectedJid, setSelectedJid] = useState('')
  const [periods, setPeriods] = useState([])
  const [status, setStatus] = useState(null)

  function loadJunction(jid) {
    setSelectedJid(jid)
    setStatus(null)
    const j = junctions.find((j) => j.junction_id === jid)
    setPeriods(j ? (j.peak_periods || []).map((p) => [...p]) : [])
  }

  function removePeriod(i) {
    setPeriods((prev) => prev.filter((_, idx) => idx !== i))
  }

  function addPeriod(p) {
    setPeriods((prev) => [...prev, p])
  }

  async function save() {
    setStatus(null)
    try {
      const res = await fetch(`${API}/admin/junctions/${selectedJid}/peak_periods`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ peak_periods: periods }),
      })
      if (!res.ok) {
        const err = await res.json()
        setStatus({ ok: false, msg: err.detail || 'Error saving' })
        return
      }
      setStatus({ ok: true, msg: 'Saved successfully' })
      onJunctionUpdate()
    } catch {
      setStatus({ ok: false, msg: 'Network error' })
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-xs text-gray-400 mb-1">Junction</label>
        <select
          value={selectedJid}
          onChange={(e) => loadJunction(e.target.value)}
          className="bg-gray-700 text-gray-200 text-sm rounded px-3 py-1.5 w-full"
        >
          <option value="">Select junction…</option>
          {junctions.map((j) => (
            <option key={j.junction_id} value={j.junction_id}>
              {j.junction_id} — {j.name}
            </option>
          ))}
        </select>
      </div>

      {selectedJid && (
        <>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Current peak periods</label>
            <PeriodChips periods={periods} onRemove={removePeriod} />
            <PeriodAdder onAdd={addPeriod} />
          </div>

          <button
            onClick={save}
            className="bg-green-700 hover:bg-green-600 text-white text-sm px-4 py-1.5 rounded"
          >
            Save Changes
          </button>

          {status && (
            <p className={`text-xs mt-1 ${status.ok ? 'text-green-400' : 'text-red-400'}`}>
              {status.msg}
            </p>
          )}
        </>
      )}
    </div>
  )
}

// ─── Tab 2: Draw Roads ───

function DrawRoadsTab({
  junctions,
  drawingArm,
  drawingPath,
  onStartDrawing,
  onStopDrawing,
  onUndoPoint,
  onClearPath,
  onJunctionUpdate,
}) {
  const [selectedJid, setSelectedJid] = useState('')
  const [selectedAid, setSelectedAid] = useState('')
  const [status, setStatus] = useState(null)

  const selectedJunction = junctions.find((j) => j.junction_id === selectedJid)
  const isDrawing = drawingArm && drawingArm.junction_id === selectedJid && drawingArm.arm_id === selectedAid

  function selectJunction(jid) {
    setSelectedJid(jid)
    setSelectedAid('')
    setStatus(null)
  }

  async function saveRoadPath() {
    setStatus(null)
    try {
      const res = await fetch(`${API}/admin/junctions/${selectedJid}/arms/${selectedAid}/road_path`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ road_path: drawingPath }),
      })
      if (!res.ok) {
        const err = await res.json()
        setStatus({ ok: false, msg: err.detail || 'Error saving' })
        return
      }
      setStatus({ ok: true, msg: `Saved ${drawingPath.length} waypoints` })
      onStopDrawing()
      onJunctionUpdate()
    } catch {
      setStatus({ ok: false, msg: 'Network error' })
    }
  }

  return (
    <div className="space-y-3">
      <div className="bg-blue-900/30 border border-blue-700 text-blue-300 text-xs rounded p-2">
        <strong>Tip:</strong> Trace the road from the far end of the approach toward the junction.
        The last point you place becomes the camera's anchor position on the map.
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Junction</label>
          <select
            value={selectedJid}
            onChange={(e) => selectJunction(e.target.value)}
            className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
          >
            <option value="">Select…</option>
            {junctions.map((j) => (
              <option key={j.junction_id} value={j.junction_id}>{j.junction_id}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Camera arm</label>
          <select
            value={selectedAid}
            onChange={(e) => setSelectedAid(e.target.value)}
            disabled={!selectedJunction}
            className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full disabled:opacity-50"
          >
            <option value="">Select…</option>
            {(selectedJunction?.arms || []).map((a) => (
              <option key={a.arm_id} value={a.arm_id}>{a.arm_id}</option>
            ))}
          </select>
        </div>
      </div>

      {selectedJid && selectedAid && (
        <>
          {!isDrawing ? (
            <button
              onClick={() => onStartDrawing(selectedJid, selectedAid)}
              className="bg-amber-600 hover:bg-amber-500 text-white text-sm px-4 py-1.5 rounded w-full"
            >
              Start Drawing
            </button>
          ) : (
            <div className="space-y-2">
              <p className="text-xs text-amber-300">
                Click on the map to place waypoints. Points placed: <strong>{drawingPath.length}</strong>
              </p>
              <div className="flex gap-2">
                <button
                  onClick={onUndoPoint}
                  disabled={drawingPath.length === 0}
                  className="bg-gray-600 hover:bg-gray-500 text-white text-xs px-3 py-1 rounded disabled:opacity-40"
                >
                  Undo Last
                </button>
                <button
                  onClick={onClearPath}
                  disabled={drawingPath.length === 0}
                  className="bg-gray-600 hover:bg-gray-500 text-white text-xs px-3 py-1 rounded disabled:opacity-40"
                >
                  Clear
                </button>
                <button
                  onClick={onStopDrawing}
                  className="bg-gray-600 hover:bg-gray-500 text-white text-xs px-3 py-1 rounded"
                >
                  Stop Drawing
                </button>
                <button
                  onClick={saveRoadPath}
                  disabled={drawingPath.length < 2}
                  className="bg-green-700 hover:bg-green-600 text-white text-xs px-3 py-1 rounded disabled:opacity-40 ml-auto"
                >
                  Save Road Path
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {status && (
        <p className={`text-xs ${status.ok ? 'text-green-400' : 'text-red-400'}`}>
          {status.msg}
        </p>
      )}
    </div>
  )
}

// ─── Tab 3: New Junction ───

function NewJunctionTab({ onJunctionUpdate }) {
  const [form, setForm] = useState({
    junction_id: '',
    name: '',
    type: '+',
    arm_id: '',
    arm_name: '',
    gps_lat: '',
    gps_lon: '',
  })
  const [periods, setPeriods] = useState([[7, 9], [17, 19]])
  const [status, setStatus] = useState(null)

  function set(key, val) {
    setForm((prev) => ({ ...prev, [key]: val }))
  }

  async function create() {
    setStatus(null)
    try {
      const res = await fetch(`${API}/admin/junctions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          junction_id: form.junction_id.trim(),
          name: form.name.trim(),
          type: form.type,
          peak_periods: periods,
          first_arm: {
            arm_id: form.arm_id.trim(),
            name: form.arm_name.trim(),
            gps_lat: parseFloat(form.gps_lat),
            gps_lon: parseFloat(form.gps_lon),
          },
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        setStatus({ ok: false, msg: data.detail || 'Error creating junction' })
        return
      }
      setStatus({ ok: true, msg: `Junction ${data.junction_id} created` })
      setForm({ junction_id: '', name: '', type: '+', arm_id: '', arm_name: '', gps_lat: '', gps_lon: '' })
      setPeriods([[7, 9], [17, 19]])
      onJunctionUpdate()
    } catch {
      setStatus({ ok: false, msg: 'Network error' })
    }
  }

  return (
    <div className="space-y-3">
      <InfoBanner text="New junctions appear on the map immediately. Live camera feeds require a server restart." />

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Junction ID</label>
          <input
            value={form.junction_id}
            onChange={(e) => set('junction_id', e.target.value)}
            placeholder="e.g. JCT03"
            className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Type</label>
          <select
            value={form.type}
            onChange={(e) => set('type', e.target.value)}
            className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
          >
            <option value="+">+ (Crossroads)</option>
            <option value="T">T (T-Junction)</option>
            <option value="L">L (L-Junction)</option>
          </select>
        </div>
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Junction Name</label>
        <input
          value={form.name}
          onChange={(e) => set('name', e.target.value)}
          placeholder="e.g. High Street / Mill Road"
          className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
        />
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Peak periods</label>
        <PeriodChips periods={periods} onRemove={(i) => setPeriods((p) => p.filter((_, idx) => idx !== i))} />
        <PeriodAdder onAdd={(p) => setPeriods((prev) => [...prev, p])} />
      </div>

      <p className="text-xs text-gray-400 font-medium mt-2">First Camera Arm</p>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Arm ID</label>
          <input
            value={form.arm_id}
            onChange={(e) => set('arm_id', e.target.value)}
            placeholder="e.g. ARM_NORTH"
            className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Arm Name</label>
          <input
            value={form.arm_name}
            onChange={(e) => set('arm_name', e.target.value)}
            placeholder="e.g. High St Northbound"
            className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">GPS Lat</label>
          <input
            value={form.gps_lat}
            onChange={(e) => set('gps_lat', e.target.value)}
            placeholder="51.5074"
            type="number"
            step="any"
            className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">GPS Lon</label>
          <input
            value={form.gps_lon}
            onChange={(e) => set('gps_lon', e.target.value)}
            placeholder="-0.1278"
            type="number"
            step="any"
            className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
          />
        </div>
      </div>

      <button
        onClick={create}
        className="bg-green-700 hover:bg-green-600 text-white text-sm px-4 py-1.5 rounded w-full mt-2"
      >
        Create Junction
      </button>

      {status && (
        <p className={`text-xs ${status.ok ? 'text-green-400' : 'text-red-400'}`}>
          {status.msg}
        </p>
      )}
    </div>
  )
}

// ─── Tab 4: Add Camera ───

function AddCameraTab({ junctions, onJunctionUpdate }) {
  const [selectedJid, setSelectedJid] = useState('')
  const [form, setForm] = useState({ arm_id: '', name: '', gps_lat: '', gps_lon: '', rtsp_url: '' })
  const [status, setStatus] = useState(null)

  function set(key, val) {
    setForm((prev) => ({ ...prev, [key]: val }))
  }

  async function addCamera() {
    setStatus(null)
    try {
      const res = await fetch(`${API}/admin/junctions/${selectedJid}/arms`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          arm_id: form.arm_id.trim(),
          name: form.name.trim(),
          gps_lat: parseFloat(form.gps_lat),
          gps_lon: parseFloat(form.gps_lon),
          rtsp_url: form.rtsp_url.trim(),
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        setStatus({ ok: false, msg: data.detail || 'Error adding camera' })
        return
      }
      setStatus({ ok: true, msg: `Camera ${data.arm_id} added to ${selectedJid}` })
      setForm({ arm_id: '', name: '', gps_lat: '', gps_lon: '', rtsp_url: '' })
      onJunctionUpdate()
    } catch {
      setStatus({ ok: false, msg: 'Network error' })
    }
  }

  return (
    <div className="space-y-3">
      <InfoBanner text="New cameras appear on the map immediately. Live video feeds require a server restart." />

      <div>
        <label className="block text-xs text-gray-400 mb-1">Junction</label>
        <select
          value={selectedJid}
          onChange={(e) => { setSelectedJid(e.target.value); setStatus(null) }}
          className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
        >
          <option value="">Select junction…</option>
          {junctions.map((j) => (
            <option key={j.junction_id} value={j.junction_id}>
              {j.junction_id} — {j.name}
            </option>
          ))}
        </select>
      </div>

      {selectedJid && (
        <>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Arm ID</label>
              <input
                value={form.arm_id}
                onChange={(e) => set('arm_id', e.target.value)}
                placeholder="e.g. ARM_SOUTH"
                className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Arm Name</label>
              <input
                value={form.name}
                onChange={(e) => set('name', e.target.value)}
                placeholder="e.g. Oak Ave Southbound"
                className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">GPS Lat</label>
              <input
                value={form.gps_lat}
                onChange={(e) => set('gps_lat', e.target.value)}
                placeholder="51.5074"
                type="number"
                step="any"
                className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">GPS Lon</label>
              <input
                value={form.gps_lon}
                onChange={(e) => set('gps_lon', e.target.value)}
                placeholder="-0.1278"
                type="number"
                step="any"
                className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">RTSP URL (optional)</label>
            <input
              value={form.rtsp_url}
              onChange={(e) => set('rtsp_url', e.target.value)}
              placeholder="rtsp://…"
              className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
            />
          </div>

          <button
            onClick={addCamera}
            className="bg-green-700 hover:bg-green-600 text-white text-sm px-4 py-1.5 rounded w-full"
          >
            Add Camera
          </button>

          {status && (
            <p className={`text-xs ${status.ok ? 'text-green-400' : 'text-red-400'}`}>
              {status.msg}
            </p>
          )}
        </>
      )}
    </div>
  )
}

// ─── Tab 5: Update Stream ───

function UpdateStreamTab({ junctions }) {
  const [selectedJid, setSelectedJid] = useState('')
  const [selectedAid, setSelectedAid] = useState('')
  const [rtspUrl, setRtspUrl] = useState('')
  const [status, setStatus] = useState(null)

  const selectedJunction = junctions.find((j) => j.junction_id === selectedJid)

  function selectJunction(jid) {
    setSelectedJid(jid)
    setSelectedAid('')
    setStatus(null)
  }

  async function updateStream() {
    setStatus(null)
    try {
      const res = await fetch(
        `${API}/admin/junctions/${selectedJid}/arms/${selectedAid}/stream`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rtsp_url: rtspUrl.trim() }),
        }
      )
      const data = await res.json()
      if (!res.ok) {
        setStatus({ ok: false, msg: data.detail || 'Error updating stream' })
        return
      }
      setStatus({ ok: true, msg: `Stream updated for ${selectedJid}/${selectedAid}` })
    } catch {
      setStatus({ ok: false, msg: 'Network error' })
    }
  }

  return (
    <div className="space-y-3">
      <InfoBanner text="Updates the RTSP URL or local file path in memory and on disk. A server restart is required to reconnect the live video feed." />

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Junction</label>
          <select
            value={selectedJid}
            onChange={(e) => selectJunction(e.target.value)}
            className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
          >
            <option value="">Select…</option>
            {junctions.map((j) => (
              <option key={j.junction_id} value={j.junction_id}>
                {j.junction_id} — {j.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Camera arm</label>
          <select
            value={selectedAid}
            onChange={(e) => { setSelectedAid(e.target.value); setStatus(null) }}
            disabled={!selectedJunction}
            className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full disabled:opacity-50"
          >
            <option value="">Select…</option>
            {(selectedJunction?.arms || []).map((a) => (
              <option key={a.arm_id} value={a.arm_id}>{a.arm_id} — {a.name}</option>
            ))}
          </select>
        </div>
      </div>

      {selectedJid && selectedAid && (
        <>
          <div>
            <label className="block text-xs text-gray-400 mb-1">New RTSP URL or file path</label>
            <input
              value={rtspUrl}
              onChange={(e) => setRtspUrl(e.target.value)}
              placeholder="rtsp://user:pass@192.168.1.10/stream  or  /path/to/video.mp4"
              className="bg-gray-700 text-gray-200 text-sm rounded px-2 py-1.5 w-full"
            />
          </div>

          <button
            onClick={updateStream}
            disabled={!rtspUrl.trim()}
            className="bg-green-700 hover:bg-green-600 text-white text-sm px-4 py-1.5 rounded w-full disabled:opacity-40"
          >
            Update Stream
          </button>
        </>
      )}

      {status && (
        <p className={`text-xs ${status.ok ? 'text-green-400' : 'text-red-400'}`}>
          {status.msg}
        </p>
      )}
    </div>
  )
}

// ─── Main AdminPanel ───

const TABS = [
  { id: 'peak', label: 'Peak Hours' },
  { id: 'roads', label: 'Draw Roads' },
  { id: 'junction', label: 'New Junction' },
  { id: 'camera', label: 'Add Camera' },
  { id: 'stream', label: 'Update Stream' },
]

export default function AdminPanel({
  junctions,
  onJunctionUpdate,
  drawingArm,
  drawingPath,
  onStartDrawing,
  onStopDrawing,
  onUndoPoint,
  onClearPath,
}) {
  const [activeTab, setActiveTab] = useState('peak')

  return (
    <div className="bg-gray-800 border border-amber-600 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-amber-900/40 border-b border-amber-700 px-4 py-2">
        <p className="text-amber-300 text-sm font-medium">
          Map editing active — drag any camera marker on the map to update its GPS position.
          Use the <strong>Draw Roads</strong> tab to trace the road shape for each camera.
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-gray-700">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-amber-400 border-b-2 border-amber-400 bg-gray-750'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {tab.label}
            {tab.id === 'roads' && drawingArm && (
              <span className="ml-1.5 bg-amber-500 text-gray-900 text-xs px-1.5 rounded-full">●</span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="p-4">
        {activeTab === 'peak' && (
          <PeakHoursTab junctions={junctions} onJunctionUpdate={onJunctionUpdate} />
        )}
        {activeTab === 'roads' && (
          <DrawRoadsTab
            junctions={junctions}
            drawingArm={drawingArm}
            drawingPath={drawingPath}
            onStartDrawing={onStartDrawing}
            onStopDrawing={onStopDrawing}
            onUndoPoint={onUndoPoint}
            onClearPath={onClearPath}
            onJunctionUpdate={onJunctionUpdate}
          />
        )}
        {activeTab === 'junction' && (
          <NewJunctionTab onJunctionUpdate={onJunctionUpdate} />
        )}
        {activeTab === 'camera' && (
          <AddCameraTab junctions={junctions} onJunctionUpdate={onJunctionUpdate} />
        )}
        {activeTab === 'stream' && (
          <UpdateStreamTab junctions={junctions} />
        )}
      </div>
    </div>
  )
}
