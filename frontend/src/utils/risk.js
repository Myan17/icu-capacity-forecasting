export function getRiskColor(riskLevel) {
    switch (riskLevel) {
      case "RED":
        return "risk-red";
      case "YELLOW":
        return "risk-yellow";
      case "GREEN":
        return "risk-green";
      default:
        return "risk-neutral";
    }
  }
  
  export function computeOccupancyRatio(occupied, capacity) {
    if (!capacity || capacity <= 0) return null;
    return occupied / capacity;
  }