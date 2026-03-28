export default function EmptyState({ message = "No data available." }) {
    return <div className="state-box">{message}</div>;
  }