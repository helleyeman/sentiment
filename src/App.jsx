import { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState("");

  const analyze = async () => {
    const res = await fetch("https://sentiment-production-b00f.up.railway.app", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    setResult(data.sentiment);
  };

  return (
    <div className="container">
      <h1>Sentiment Analyzer</h1>
      <textarea
        rows="5"
        placeholder="Type something..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button onClick={analyze}>Analyze</button>
      {result && <div className="result">Result: {result}</div>}
    </div>
  );
}

export default App;
