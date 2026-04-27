import axios from "axios";

const API_BASE_URL = "http://52.15.187.251:8000";

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

export async function runForecast(hospitalId, model = null) {
  const params = model ? `?model=${model}` : "";
  const response = await api.post(`/forecast/${hospitalId}${params}`);
  return response.data;
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