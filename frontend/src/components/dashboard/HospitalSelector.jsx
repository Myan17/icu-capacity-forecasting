import { useEffect, useState } from "react";
import { getHospitals } from "../../api/hospitalApi";

export default function HospitalSelector({ value, onChange }) {
  const [hospitals, setHospitals] = useState(["010001"]);

  useEffect(() => {
    getHospitals()
      .then(setHospitals)
      .catch(() => {});
  }, []);

  return (
    <div className="hospital-selector">
      <label htmlFor="hospital-select">Hospital</label>
      <select
        id="hospital-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {hospitals.map((id) => (
          <option key={id} value={id}>{id}</option>
        ))}
      </select>
    </div>
  );
}
