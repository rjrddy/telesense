import { useState, useEffect } from 'react'
import './KeyGate.css'

/**
 * Thin wrapper over the demo-key stored in localStorage. Shows a small input
 * strip at the top of the app so visitors can paste the shared key once and
 * forget it. Works with local and deployed backends.
 */
export default function KeyGate() {
  const [key, setKey] = useState(() => {
    try { return localStorage.getItem('telesense_demo_key') || '' } catch { return '' }
  })
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setSaved(false), 1500)
    return () => clearTimeout(t)
  }, [saved])

  const save = (e) => {
    e.preventDefault()
    try { localStorage.setItem('telesense_demo_key', key) } catch { /* ignore */ }
    setSaved(true)
  }

  return (
    <form className="key-gate" onSubmit={save}>
      <label htmlFor="demo-key">demo key</label>
      <input
        id="demo-key"
        type="password"
        value={key}
        onChange={(e) => setKey(e.target.value)}
        placeholder="paste key…"
        autoComplete="off"
      />
      <button type="submit">{saved ? 'saved' : 'save'}</button>
    </form>
  )
}
