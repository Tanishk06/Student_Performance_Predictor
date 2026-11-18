import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const groups = {
  "Academic Information": [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Tutoring_Sessions",
  ],
  "Learning Environment": [
    "Access_to_Resources",
    "Teacher_Quality",
    "School_Type",
    "Internet_Access",
    "Peer_Influence",
  ],
  "Personal Routine": [
    "Sleep_Hours",
    "Physical_Activity",
    "Motivation_Level",
    "Extracurricular_Activities",
  ],
  "Family Background": [
    "Parental_Involvement",
    "Parental_Education_Level",
    "Family_Income",
    "Distance_from_Home",
  ],
  "Other Information": ["Learning_Disabilities", "Gender"],
};

function formatLabel(name) {
  return name.replaceAll("_", " ");
}

function App() {
  const [schema, setSchema] = useState(null);
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loadingSchema, setLoadingSchema] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  // Fetch the model's input schema from FastAPI.
  useEffect(() => {
    async function loadSchema() {
      try {
        const response = await fetch(`${API_URL}/schema`);

        if (!response.ok) {
          throw new Error("Could not load the model input schema.");
        }

        const data = await response.json();

        const initialValues = {};

        data.features.forEach((feature) => {
          if (data.numeric_features[feature]) {
            initialValues[feature] = data.numeric_features[feature].median;
          } else {
            initialValues[feature] = data.categorical_features[feature][0];
          }
        });

        setSchema(data);
        setFormData(initialValues);
      } catch (err) {
        setError(
          "Unable to connect to the prediction server. Make sure FastAPI is running.",
        );
      } finally {
        setLoadingSchema(false);
      }
    }

    loadSchema();
  }, []);

  function handleChange(feature, value, isNumeric) {
    setFormData((previous) => ({
      ...previous,
      [feature]: isNumeric ? Number(value) : value,
    }));

    setPrediction(null);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setPrediction(null);
    setSubmitting(true);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof result.detail === "string"
            ? result.detail
            : JSON.stringify(result.detail),
        );
      }

      setPrediction(result);
    } catch (err) {
      setError(err.message || "Prediction failed. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  if (loadingSchema) {
    return (
      <main className="page">
        <p className="loading">Loading prediction system...</p>
      </main>
    );
  }

  if (!schema) {
    return (
      <main className="page">
        <div className="error-box">{error}</div>
      </main>
    );
  }

  return (
    <main className="page">
      <header className="hero">
        <span className="eyebrow">MACHINE LEARNING PROJECT</span>
        <h1>Student Performance Predictor</h1>
        <p>
          Enter student information to receive an estimated examination score
          powered by machine learning.
        </p>
        <div className="model-tag">Model: {schema.model_name}</div>
      </header>

      <form className="form-card" onSubmit={handleSubmit}>
        {Object.entries(groups).map(([groupName, fields]) => (
          <section className="form-section" key={groupName}>
            <div className="section-heading">
              <h2>{groupName}</h2>
            </div>

            <div className="form-grid">
              {fields
                .filter((feature) => schema.features.includes(feature))
                .map((feature) => {
                  const numeric = schema.numeric_features[feature];
                  const options = schema.categorical_features[feature];

                  return (
                    <div className="field" key={feature}>
                      <label htmlFor={feature}>{formatLabel(feature)}</label>

                      {numeric ? (
                        <>
                          <input
                            id={feature}
                            type="number"
                            min={numeric.min}
                            max={numeric.max}
                            step="1"
                            value={formData[feature] ?? ""}
                            required
                            onChange={(e) =>
                              handleChange(feature, e.target.value, true)
                            }
                          />
                          <small>
                            Range: {numeric.min}–{numeric.max}
                          </small>
                        </>
                      ) : (
                        <select
                          id={feature}
                          value={formData[feature] ?? ""}
                          required
                          onChange={(e) =>
                            handleChange(feature, e.target.value, false)
                          }
                        >
                          {options.map((option) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>
                  );
                })}
            </div>
          </section>
        ))}

        <button className="predict-button" type="submit" disabled={submitting}>
          {submitting ? "Predicting..." : "Predict Exam Score"}
        </button>

        {error && <div className="error-box">{error}</div>}
      </form>

      {prediction && (
        <section className="result-card">
          <span className="eyebrow">PREDICTION RESULT</span>
          <h2>{prediction.predicted_exam_score}</h2>
          <p>Estimated Exam Score</p>
          <div className="result-model">Model: {prediction.model}</div>
          <p className="disclaimer">{prediction.notice}</p>
        </section>
      )}

      <footer>Student Performance Prediction · Machine Learning Project</footer>
    </main>
  );
}

export default App;
