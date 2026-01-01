import React, { useState } from 'react';
import axios from 'axios';
import { Wind, Thermometer, Droplets, Activity, Info, AlertTriangle, CheckCircle } from 'lucide-react';
import { RadialBarChart, RadialBar, Legend, Tooltip, ResponsiveContainer } from 'recharts';

function App() {
  const [formData, setFormData] = useState({
    // Defaults (approximate means)
    ozone_lag_1: 40,
    tmax: 25,
    tavg: 20,
    CUTI: 0.5,
    month_sin: 0.5,
    month_cos: 0.5,
    wspd: 10,
    Tmax_inland: 30,
    land_sea_temp_diff: 5,
    CUTI_lag1: 0.5,
    CUTI_lag3: 0.5,
    CUTI_roll7_mean: 0.5,
    thermal_stability: 0.1,
    marine_layer_presence: 0,
    BEUTI: 10,
    tsun: 10,
    temp_range: 10,
    CUTI_lag7: 0.5,
    sst_value_sst: 18,
    sst_anomaly: 0,
    distance_to_coast_km: 5
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [logs, setLogs] = useState([]);

  const fetchLogs = async () => {
    try {
      // Truncate trailing slash if present to avoid //api/... (protocol relative) bug if user sets VITE_API_URL to '/'
      let envUrl = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '');
      const apiUrl = envUrl.replace(/\/$/, '');
      const response = await axios.get(`${apiUrl}/api/logs/`);
      setLogs(response.data);
    } catch (err) {
      console.error("Failed to fetch logs", err);
    }
  };

  React.useEffect(() => {
    fetchLogs();
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: parseFloat(e.target.value)
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      let envUrl = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '');
      const apiUrl = envUrl.replace(/\/$/, '');
      const response = await axios.post(`${apiUrl}/api/predict/`, formData);
      setPrediction(response.data.prediction);
      fetchLogs();
    } catch (err) {
      setError('Failed to get prediction. Ensure backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getAQIColor = (category) => {
    switch (category) {
      case 'Good': return 'text-green-400';
      case 'Moderate': return 'text-yellow-400';
      case 'Unhealthy for Sensitive Groups': return 'text-orange-400';
      case 'Unhealthy': return 'text-red-500';
      case 'Very Unhealthy': return 'text-purple-500';
      case 'Hazardous': return 'text-red-900';
      default: return 'text-white';
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-50 p-8 font-sans selection:bg-ocean-500 selection:text-white">
      <div className="max-w-6xl mx-auto">

        {/* Header */}
        <header className="mb-12 text-center">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-ocean-300 to-ocean-600 bg-clip-text text-transparent mb-4">
            Ozone Ocean Predictor
          </h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            Harnessing the power of Ocean Upwelling (CUTI/BEUTI) and Meteorology to forecast coastal air quality in San Diego.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* Input Panel */}
          <div className="lg:col-span-1 bg-slate-800/50 p-6 rounded-2xl border border-slate-700 backdrop-blur-sm shadow-xl">
            <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2 text-ocean-300">
              <Activity className="w-6 h-6" />
              Model Inputs
            </h2>

            <form onSubmit={handleSubmit} className="space-y-6">

              {/* Key Drivers */}
              <div className="space-y-4">
                <h3 className="text-sm uppercase tracking-wider text-slate-500 font-bold">Key Drivers</h3>

                <div>
                  <label className="block text-sm font-medium mb-1 flex justify-between">
                    <span>Max Temperature (°C)</span>
                    <span className="text-ocean-400">{formData.tmax}</span>
                  </label>
                  <input
                    type="range" name="tmax" min="10" max="45" step="0.1"
                    value={formData.tmax} onChange={handleChange}
                    className="w-full accent-ocean-500 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1 flex justify-between">
                    <span>Upwelling Index (CUTI)</span>
                    <span className="text-ocean-400">{formData.CUTI}</span>
                  </label>
                  <input
                    type="range" name="CUTI" min="-1" max="3" step="0.1"
                    value={formData.CUTI} onChange={handleChange}
                    className="w-full accent-ocean-500 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                  />
                  <p className="text-xs text-slate-500 mt-1">Higher values = Stronger Upwelling (Cooling)</p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1 flex justify-between">
                    <span>Wind Speed (km/h)</span>
                    <span className="text-ocean-400">{formData.wspd}</span>
                  </label>
                  <input
                    type="range" name="wspd" min="0" max="50" step="0.5"
                    value={formData.wspd} onChange={handleChange}
                    className="w-full accent-ocean-500 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1 flex justify-between">
                    <span>SST (°C)</span>
                    <span className="text-ocean-400">{formData.sst_value_sst}</span>
                  </label>
                  <input
                    type="range" name="sst_value_sst" min="10" max="30" step="0.1"
                    value={formData.sst_value_sst} onChange={handleChange}
                    className="w-full accent-ocean-500 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>

              {/* Advanced Inputs (Hidden/Collapsed or just less prominent) */}
              <div className="pt-4 border-t border-slate-700">
                <h3 className="text-sm uppercase tracking-wider text-slate-500 font-bold mb-4">Secondary Factors</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Ozone Lag 1</label>
                    <input type="number" name="ozone_lag_1" value={formData.ozone_lag_1} onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm focus:border-ocean-500 outline-none" />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Inland Tmax</label>
                    <input type="number" name="Tmax_inland" value={formData.Tmax_inland} onChange={handleChange} className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm focus:border-ocean-500 outline-none" />
                  </div>
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-ocean-600 hover:bg-ocean-500 text-white font-bold py-3 px-4 rounded-xl transition-all shadow-lg shadow-ocean-900/50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Calculating...' : 'Predict Ozone Level'}
              </button>
            </form>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 space-y-8">

            {/* Main Prediction Card */}
            <div className="bg-slate-800/50 p-8 rounded-2xl border border-slate-700 backdrop-blur-sm shadow-xl min-h-[400px] flex flex-col justify-center items-center relative overflow-hidden">
              {!prediction && !loading && (
                <div className="text-center text-slate-500">
                  <Wind className="w-16 h-16 mx-auto mb-4 opacity-20" />
                  <p className="text-xl">Adjust inputs and click Predict to see the forecast.</p>
                </div>
              )}

              {loading && (
                <div className="animate-pulse text-center">
                  <div className="w-32 h-32 rounded-full bg-slate-700 mx-auto mb-4"></div>
                  <div className="h-4 bg-slate-700 w-48 mx-auto rounded"></div>
                </div>
              )}

              {prediction && (
                <div className="w-full animate-in fade-in zoom-in duration-500">
                  <div className="text-center mb-8">
                    <h3 className="text-slate-400 text-lg uppercase tracking-widest mb-2">Predicted Ozone Level</h3>
                    <div className="text-8xl font-black tracking-tighter text-white mb-2">
                      {prediction.predicted_ozone.toFixed(1)} <span className="text-3xl font-normal text-slate-500">ppb</span>
                    </div>
                    <div className={`text-3xl font-bold ${getAQIColor(prediction.air_quality_category)}`}>
                      {prediction.air_quality_category}
                    </div>
                  </div>

                  {/* Regime Badge */}
                  <div className="bg-slate-900/80 rounded-xl p-6 border border-slate-700 max-w-lg mx-auto">
                    <div className="flex items-start gap-4">
                      <div className="p-3 bg-ocean-900/50 rounded-lg text-ocean-400">
                        <Info className="w-6 h-6" />
                      </div>
                      <div>
                        <h4 className="text-lg font-semibold text-ocean-200 mb-1">Detected Regime</h4>
                        <p className="text-white font-medium text-xl mb-2">{prediction.regime_description}</p>
                        <p className="text-sm text-slate-400">
                          Based on current upwelling (CUTI) and temperature gradients, the model has identified this specific atmospheric regime.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Explanation / Context */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-800/30 p-6 rounded-xl border border-slate-700/50">
                <h3 className="font-semibold text-slate-300 mb-2 flex items-center gap-2">
                  <Thermometer className="w-4 h-4" /> Why this prediction?
                </h3>
                <p className="text-sm text-slate-400 leading-relaxed">
                  High temperatures usually increase Ozone, but strong upwelling (High CUTI) can bring cool marine air that disrupts ozone formation. The model balances these competing forces.
                </p>
              </div>
              <div className="bg-slate-800/30 p-6 rounded-xl border border-slate-700/50">
                <h3 className="font-semibold text-slate-300 mb-2 flex items-center gap-2">
                  <Droplets className="w-4 h-4" /> Ocean Influence
                </h3>
                <p className="text-sm text-slate-400 leading-relaxed">
                  The "Ozone-Ocean" hypothesis suggests that biological upwelling (BEUTI) and physical transport (CUTI) are critical predictors for coastal San Diego.
                </p>
              </div>
            </div>

          </div>
        </div>

        {/* Database History Log */}
        <div className="mt-16 border-t border-slate-700 pt-12">
          <h2 className="text-2xl font-semibold mb-6 text-ocean-300">Database History Log</h2>
          <div className="bg-slate-800/50 rounded-xl overflow-hidden border border-slate-700">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-slate-400">
                <thead className="bg-slate-700/50 text-slate-200 uppercase tracking-wider font-semibold">
                  <tr>
                    <th className="p-4">ID</th>
                    <th className="p-4">Time</th>
                    <th className="p-4">Input (Partial)</th>
                    <th className="p-4">Prediction</th>
                    <th className="p-4">Source</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700">
                  {logs.map((log) => (
                    <tr key={log.id} className="hover:bg-slate-700/30 transition-colors">
                      <td className="p-4 font-mono text-xs text-slate-500">#{log.id}</td>
                      <td className="p-4">{new Date(log.created_at).toLocaleString()}</td>
                      <td className="p-4">
                        <span className="font-mono text-xs">
                          Tmax: {log.input.tmax} | CUTI: {log.input.CUTI} | ...
                        </span>
                      </td>
                      <td className="p-4 text-white">
                        {log.output.predicted_ozone
                          ? `${log.output.predicted_ozone.toFixed(1)} ppb (${log.output.air_quality_category})`
                          : log.output.prediction_score ? `Score: ${log.output.prediction_score}` : 'N/A'}
                      </td>
                      <td className="p-4">
                        <span className={`px-2 py-1 rounded text-xs font-bold ${log.output.source === 'ML Model' ? 'bg-green-900 text-green-300' : 'bg-yellow-900 text-yellow-300'}`}>
                          {log.output.source || 'Mock'}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {logs.length === 0 && (
                    <tr>
                      <td colSpan="5" className="p-8 text-center text-slate-500">
                        No history found. Make a prediction to see it here.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
