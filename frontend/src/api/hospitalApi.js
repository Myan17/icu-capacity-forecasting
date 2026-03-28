import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

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

export async function runForecast(hospitalId) {
  const response = await api.post(`/forecast/${hospitalId}`);
  return response.data;
}

export async function getAlerts(hospitalId) {
  const response = await api.get(`/alerts/${hospitalId}`);
  return response.data;
}