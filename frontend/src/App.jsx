import React, { useState, useRef, useCallback, useEffect } from 'react'
import { predictDisease, checkHealth } from './api.js'

export default function App() {
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [status, setStatus] = useState('idle') // idle | analyzing | done | error
  const [result, setResult] = useState(null)
  const [errorMsg, setErrorMsg] = useState('')
  const [backendUp, setBackendUp] = useState(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    checkHealth().then(setBackendUp)
  }, [])

  const handleFile = useCallback((file) => {
   const isImageType = file &&        (file.type.startsWith('image/') || /\. (heic|heif|webp|bmp|gif)$/i.test(file.name))
    if (!isImageType) return
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    setStatus('idle')
    setResult(null)
    setErrorMsg('')
  }, [])

  const onDrop = (e) => {
    e.preventDefault()
    handleFile(e.dataTransfer.files[0])
  }

  const onSelect = (e) => handleFile(e.target.files[0])

  const runScan = async () => {
    if (!imageFile) return
    setStatus('analyzing')
    setErrorMsg('')
    try {
      const data = await predictDisease(imageFile)
      setResult(data)
      setStatus('done')
    } catch (err) {
      setErrorMsg(err.message || 'Something went wrong reaching the scanner.')
      setStatus('error')
    }
  }

  const reset = () => {
    setImageFile(null)
    setImagePreview(null)
    setStatus('idle')
    setResult(null)
    setErrorMsg('')
  }

  return (
    <div className="page">
      <div className="field-texture" aria-hidden="true" />

      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">कि</span>
          <span className="brand-name">KissanConnect</span>
        </div>
        <div className={`backend-pill ${backendUp ? 'up' : backendUp === false ? 'down' : ''}`}>
          <span className="dot" />
          {backendUp === null ? 'checking scanner…' : backendUp ? 'scanner online' : 'scanner offline'}
        </div>
      </header>

      <main className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Leaf diagnosis, 38 conditions, 14 crops</p>
          <h1>Point your camera at the leaf.<br />We'll read what it's telling you.</h1>
          <p className="lede">
            Trained on the PlantVillage dataset with a MobileNetV2 vision model.
            Upload one clear photo of the affected leaf to get a diagnosis and
            a confidence score in seconds.
          </p>
        </div>

        <div className="scan-panel">
          <div
            className={`scan-ring ${status}`}
            onDrop={onDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => !imagePreview && fileInputRef.current?.click()}
          >
            <svg className="ring-svg" viewBox="0 0 300 300" aria-hidden="true">
              <circle className="ring-track" cx="150" cy="150" r="130" />
              <circle className="ring-sweep" cx="150" cy="150" r="130" />
              {[...Array(24)].map((_, i) => {
                const angle = (i / 24) * 2 * Math.PI
                const x1 = 150 + 118 * Math.cos(angle)
                const y1 = 150 + 118 * Math.sin(angle)
                const x2 = 150 + 130 * Math.cos(angle)
                const y2 = 150 + 130 * Math.sin(angle)
                return <line key={i} className="vein-tick" x1={x1} y1={y1} x2={x2} y2={y2} />
              })}
            </svg>

            {imagePreview ? (
              <img src={imagePreview} alt="Uploaded leaf" className="preview-img" />
            ) : (
              <div className="drop-hint">
                <span className="drop-icon">🍃</span>
                <span>Drop a leaf photo<br />or tap to choose one</span>
              </div>
            )}

            {status === 'analyzing' && <div className="analyzing-label">reading leaf…</div>}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/png, image/jpeg, image/webp, image/heic, image/heif, image/bmp, image/gif, .heic, .heif"
            onChange={onSelect}
            hidden
          />

          <div className="controls">
            {imagePreview && status !== 'analyzing' && (
              <>
                <button className="btn btn-primary" onClick={runScan} disabled={status === 'analyzing'}>
                  Scan this leaf
                </button>
                <button className="btn btn-ghost" onClick={reset}>Choose another photo</button>
              </>
            )}
            {!imagePreview && (
              <button className="btn btn-primary" onClick={() => fileInputRef.current?.click()}>
                Choose a photo
              </button>
            )}
          </div>

          {status === 'error' && (
            <div className="error-box">
              <strong>Couldn't complete the scan.</strong>
              <p>{errorMsg}</p>
              <p className="error-hint">Check that the Flask server is running at localhost:5000.</p>
            </div>
          )}
        </div>
      </main>

      {status === 'done' && result && <Results result={result} />}

      <footer className="foot">
        <span>KissanConnect — Phase 3 · React frontend, Flask API, MobileNetV2 model</span>
      </footer>
    </div>
  )
}

function Results({ result }) {
  const { prediction, top_3 } = result
  const pct = Math.round(prediction.confidence * 100)

  return (
    <section className="results">
      <div className={`result-card ${prediction.is_healthy ? 'healthy' : 'diseased'}`}>
        <div className="result-main">
          <p className="result-crop">{prediction.crop}</p>
          <h2 className="result-condition">
            {prediction.is_healthy ? 'Looks healthy' : prediction.condition}
          </h2>
          <div className="confidence-row">
            <span className="confidence-num">{pct}%</span>
            <span className="confidence-label">confidence</span>
          </div>
          <div className="confidence-bar">
            <div className="confidence-fill" style={{ width: `${pct}%` }} />
          </div>
        </div>
        <div className="result-status-icon" aria-hidden="true">
          {prediction.is_healthy ? '✓' : '⚠'}
        </div>
      </div>

      <div className="other-possibilities">
        <p className="op-label">Other possibilities considered</p>
        <ul className="op-list">
          {top_3.slice(1).map((item, i) => (
            <li key={i}>
              <span className="op-name">{item.crop} — {item.is_healthy ? 'Healthy' : item.condition}</span>
              <span className="op-pct">{Math.round(item.confidence * 100)}%</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
