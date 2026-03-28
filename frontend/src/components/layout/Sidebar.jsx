import { NavLink } from "react-router-dom";

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-title">ICU Forecast</div>

      <nav className="sidebar-nav">
        <NavLink
          to="/"
          className={({ isActive }) =>
            isActive ? "nav-item active" : "nav-item"
          }
        >
          Dashboard
        </NavLink>

        <NavLink
          to="/analytics"
          className={({ isActive }) =>
            isActive ? "nav-item active" : "nav-item"
          }
        >
          Analytics
        </NavLink>

        <NavLink
          to="/alerts"
          className={({ isActive }) =>
            isActive ? "nav-item active" : "nav-item"
          }
        >
          Alerts
        </NavLink>

        <NavLink
          to="/settings"
          className={({ isActive }) =>
            isActive ? "nav-item active" : "nav-item"
          }
        >
          Settings
        </NavLink>
      </nav>
    </aside>
  );
}