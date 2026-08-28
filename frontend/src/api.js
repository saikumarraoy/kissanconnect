// Single source of truth for the backend URL.
// Locally this falls back to localhost:5000. In production (Vercel), set
// the VITE_API_URL environment variable to your deployed Render URL,
// e.g. https://kissanconnect-api.onrender.com
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

export async function predictDisease(imageFile) {
  const formData = new FormData()
  formData.append('image', imageFile)

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const errBody = await response.json().catch(() => ({}))
    throw new Error(errBody.error || `Request failed with status ${response.status}`)
  }

  return response.json()
}

export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    return response.ok
  } catch {
    return false
  }
}
