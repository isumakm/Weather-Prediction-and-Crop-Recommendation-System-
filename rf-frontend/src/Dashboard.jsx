import { useState } from "react";
import "./Dashboard.css";

const API = "http://127.0.0.1:5000";

export default function WeatherPrediction() {
  const [form, setForm] = useState({
    date: "",
    season: "NE-Monsoon",
    location_id: "",
  });

  const [result, setResult] = useState(null); // {temperature_C, rainfall_mm, sunshine_h}
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const seasons = ["NE-Monsoon", "SW-Monsoon", "Inter-Monsoon-1", "Inter-Monsoon-2"];

  const onChange = (key, value) => setForm((p) => ({ ...p, [key]: value }));

  const onPredict = async () => {
    setError("");
    setResult(null);

    if (!form.date) return setError("Please select a date.");
    if (!form.location_id) return setError("Please enter location_id.");

    setLoading(true);
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

      setResult({
        temperature_C: data.temperature_C ?? data.prediction,
        rainfall_mm: data.rainfall_mm,
        sunshine_h: data.sunshine_h,
      });
    } catch {
      setError("Cannot connect to Flask. Make sure Flask is running on 127.0.0.1:5000");
    } finally {
      setLoading(false);
    }
  };

  const onCropRecommendation = () => {
    // For now: you can navigate later / call another page/API
    alert("Crop Recommendation button clicked ✅ (connect next page/api here)");
  };

  return (
    <div className="wp">
      <header className="header">
  <div className="header-content">
    <h1>Weather Prediction and Crop Recommendation System</h1>
    <p>Western Province, Sri Lanka</p>
  </div>

  <div className="header-controls">
    <button className="auth-button">Sign In</button>
  </div>
</header>

      <main className="wpContainer">
        <div className="sectionHeader">
  <span className="sectionBar" />
  <span className="sectionIcon">🌦️</span>
  <h3 className="sectionTitle">Weather Prediction</h3>

  {/* optional right tag like "SYSTEM OUTPUT" */}
  <span className="sectionTag">SYSTEM OUTPUT</span>
</div>
        <section className="wpCard">
          <div className="wpCardHeader">
            <h2>Weather Data</h2>
           
          </div>

          {error && <div className="wpError">{error}</div>}

          <div className="wpGrid">
            <div className="wpField">
              <label>Date</label>
              <input
                type="date"
                value={form.date}
                onChange={(e) => onChange("date", e.target.value)}
              />
            </div>

            <div className="wpField">
              <label>Season</label>
              <select value={form.season} onChange={(e) => onChange("season", e.target.value)}>
                {seasons.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            <div className="wpField">
              <label>Location ID</label>
              <input
                type="text"
                placeholder="e.g., 1"
                value={form.location_id}
                onChange={(e) => onChange("location_id", e.target.value)}
              />
            </div>
          </div>

          <button className="wpBtn" onClick={onPredict} disabled={loading}>
            {loading ? "Predicting..." : "Predict Weather"}
          </button>

          <div className="wpDivider" />

          {/* Results */}
          <div className="wpResultsGrid">
            <div className="wpResultCard">
              <div className="wpResultTitle">Temperature</div>
              <div className="wpResultValue">
                {result ? `${result.temperature_C} °C` : "—"}
              </div>
            </div>

            <div className="wpResultCard">
              <div className="wpResultTitle">Rainfall</div>
              <div className="wpResultValue">
                {result ? `${Number(result.rainfall_mm).toFixed(2)} mm` : "—"}
              </div>
            </div>

            <div className="wpResultCard">
              <div className="wpResultTitle">Sunshine</div>
              <div className="wpResultValue">
                {result ? `${Number(result.sunshine_h).toFixed(1)} h` : "—"}
              </div>
            </div>
          </div>

          {/* Crop Recommendation Button (only after prediction) */}
          {result && (
            <button className="wpBtnSecondary" onClick={onCropRecommendation}>
              Crop Recommendation
            </button>
          )}
        </section>
      </main>
    </div>
  );
}