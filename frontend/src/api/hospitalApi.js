import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export async function getLatestSnapshot(hospitalId) {
  const response = await api.get(`/snapshots/latest/${hospitalId}`);
  return response.data;
}

export async function getForecasts(hospitalId) {
  const response = await api.get(`/forecasts/${hospitalId}`);
  return response.data;
}

export async function triggerForecast(hospitalId, model = null) {
  const params = model ? `?model=${model}` : "";
  const response = await api.post(`/forecast/${hospitalId}${params}`);
  return response.data; // {status: "queued", hospital_id}
}

// Polls GET /forecasts until a result with a different run_id appears or timeout elapses.
// Returns the new forecasts array on success, or null on timeout.
export async function pollForNewForecast(
  hospitalId,
  priorRunId,
  { intervalMs = 2000, timeoutMs = 30000 } = {}
) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, intervalMs));
    const data = await getForecasts(hospitalId);
    if (data.length > 0 && data[0].run_id !== priorRunId) {
      return data;
    }
  }
  return null;
}

export async function getAlerts(hospitalId) {
  const response = await api.get(`/alerts/${hospitalId}`);
  return response.data;
}

export async function getHospitals() {
  const response = await api.get("/hospitals");
  return response.data;
}

export async function getAllAlerts(limit = 50) {
  const response = await api.get(`/alerts?limit=${limit}`);
  return response.data;
}

export async function getLoadTestLatest() {
  const response = await api.get("/load-test/latest");
  return response.data;
}

export async function getLoadTestRuns() {
  const response = await api.get("/load-test/runs");
  return response.data;
}