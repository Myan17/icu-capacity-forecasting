export function formatNumber(value) {
    if (value === null || value === undefined) return "--";
    return Number(value).toLocaleString();
  }
  
  export function formatPercent(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "--";
    return `${(value * 100).toFixed(1)}%`;
  }
  
  export function formatDateTime(value) {
    if (!value) return "--";
    return new Date(value).toLocaleString();
  }