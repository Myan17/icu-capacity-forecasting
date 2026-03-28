export default function LoadingState({ message = "Loading..." }) {
    return <div className="state-box">{message}</div>;
  }