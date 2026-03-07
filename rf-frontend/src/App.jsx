import { useState } from "react";
import Dashboard from "./Dashboard";

const API = "http://127.0.0.1:5000";

export default function App() {
  const [form, setForm] = useState({
    date: "",
    season: "NE-Monsoon",
    location_id: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");

  const seasons = ["NE-Monsoon", "SW-Monsoon", "Inter-Monsoon-1", "Inter-Monsoon-2"];

  const onChange = (key, value) => setForm((p) => ({ ...p, [key]: value }));

  const predict = async () => {
    setError("");
    setPrediction(null);

    if (!form.date) return setError("Please select a date.");
    if (!form.location_id) return setError("Please enter location_id.");

    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data?.error || "Prediction failed.");
        return;
      }

      setPrediction(data.prediction);
    } catch (e) {
      setError("Cannot connect to Flask. Make sure Flask is running on 127.0.0.1:5000");
    }
  };

   return <Dashboard />;
}