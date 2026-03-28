export default function ErrorState({ message = "Something went wrong." }) {
    return <div className="state-box error-state">{message}</div>;
  }