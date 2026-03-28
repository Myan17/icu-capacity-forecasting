export default function HospitalSelector({ value, onChange }) {
    return (
      <div className="hospital-selector">
        <label htmlFor="hospital-select">Hospital</label>
        <select
          id="hospital-select"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="HOSPITAL_A">HOSPITAL_A</option>
        </select>
      </div>
    );
  }