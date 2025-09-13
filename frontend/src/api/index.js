import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1', // proxied to Flask
  headers: { 'Content-Type': 'application/json' }
})

export async function chatOnce(payload) {
  const res = await api.post('/chat/completions', payload)
  return res.data
}
