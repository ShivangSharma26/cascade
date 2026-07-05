import React, { useState } from 'react';
import axios from 'axios';
import { Activity, Zap, BarChart2, ChevronRight, LayoutDashboard, Crosshair, Cpu } from 'lucide-react';

function App() {
  const [formData, setFormData] = useState({
    user_id: 'U00001',
    ad_id: 'A0001',
    campaign_id: 'C001',
    category: 'electronics',
    device_type: 'mobile',
    age_group: '25-34',
    hour: 20,
    day_of_week: 5,
    position: 1
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await axios.post('http://localhost:8000/predict', formData);
      setResult(response.data);
    } catch (err) {
      setError('Failed to connect to the backend API. Is the server running?');
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen relative overflow-hidden font-sans">
      {/* Deep Space Background with Radial Gradients */}
      <div className="absolute inset-0 bg-[#0a0a0a] z-[-2]"></div>
      <div className="absolute top-[-20%] left-[-10%] w-[70vw] h-[70vw] rounded-full bg-neon-red/5 blur-[120px] z-[-1]"></div>
      <div className="absolute bottom-[-20%] right-[-10%] w-[50vw] h-[50vw] rounded-full bg-neon-red/5 blur-[100px] z-[-1]"></div>

      {/* Navigation Header */}
      <nav className="glass-panel mx-4 mt-4 mb-8 px-6 py-4 flex items-center justify-between sticky top-4 z-50">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-neon-red to-red-900 flex items-center justify-center neon-glow">
            <Activity className="text-white" size={24} />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">CASCADE <span className="text-gray-500 font-normal">CTR Engine</span></h1>
        </div>
        <a 
          href="http://localhost:5000" 
          target="_blank" 
          rel="noreferrer"
          className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-all duration-300 border border-white/10 text-gray-300 hover:text-white"
        >
          <LayoutDashboard size={18} />
          <span>MLflow Registry</span>
        </a>
      </nav>

      <main className="max-w-6xl mx-auto px-4 grid grid-cols-1 lg:grid-cols-12 gap-8 pb-12">
        
        {/* Left Column - Input Form */}
        <div className="lg:col-span-8 space-y-6">
          <div className="glass-panel p-8 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-100">
            <div className="flex items-center space-x-3 mb-6">
              <Crosshair className="text-neon-red" size={24} />
              <h2 className="text-xl font-semibold text-white tracking-tight">Real-Time Impression Data</h2>
            </div>
            
            <form onSubmit={handlePredict} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-400">User ID</label>
                  <input type="text" name="user_id" value={formData.user_id} onChange={handleChange} className="w-full bg-black/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-neon-red/50 focus:ring-1 focus:ring-neon-red/50 transition-all" />
                </div>
                
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-400">Ad ID</label>
                  <input type="text" name="ad_id" value={formData.ad_id} onChange={handleChange} className="w-full bg-black/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-neon-red/50 focus:ring-1 focus:ring-neon-red/50 transition-all" />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-400">Category</label>
                  <select name="category" value={formData.category} onChange={handleChange} className="w-full bg-black/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-neon-red/50 focus:ring-1 focus:ring-neon-red/50 transition-all appearance-none">
                    <option value="electronics">Electronics</option>
                    <option value="fashion">Fashion</option>
                    <option value="home">Home & Garden</option>
                    <option value="sports">Sports</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-400">Device Type</label>
                  <select name="device_type" value={formData.device_type} onChange={handleChange} className="w-full bg-black/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-neon-red/50 focus:ring-1 focus:ring-neon-red/50 transition-all appearance-none">
                    <option value="mobile">Mobile</option>
                    <option value="desktop">Desktop</option>
                    <option value="tablet">Tablet</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-400">Hour of Day (0-23)</label>
                  <input type="number" min="0" max="23" name="hour" value={formData.hour} onChange={handleChange} className="w-full bg-black/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-neon-red/50 focus:ring-1 focus:ring-neon-red/50 transition-all" />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-400">Ad Position (1-10)</label>
                  <input type="number" min="1" max="10" name="position" value={formData.position} onChange={handleChange} className="w-full bg-black/50 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-neon-red/50 focus:ring-1 focus:ring-neon-red/50 transition-all" />
                </div>

              </div>

              <button 
                type="submit" 
                disabled={loading}
                className="w-full mt-8 bg-neon-red hover:bg-red-600 text-white font-semibold rounded-lg px-6 py-4 flex items-center justify-center space-x-2 transition-all duration-300 neon-glow hover:neon-glow-strong disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <span className="flex items-center space-x-2">
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    <span>Processing Inference...</span>
                  </span>
                ) : (
                  <>
                    <span>Execute CTR Prediction</span>
                    <ChevronRight size={20} />
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Right Column - Results Panel */}
        <div className="lg:col-span-4">
          <div className="glass-panel p-8 h-full flex flex-col items-center justify-center min-h-[400px] animate-in fade-in slide-in-from-bottom-8 duration-700 delay-300">
            
            {!result && !error && !loading && (
              <div className="text-center space-y-4 text-gray-500">
                <Cpu size={48} className="mx-auto opacity-50" />
                <p>Awaiting payload to initialize inference engine.</p>
              </div>
            )}

            {error && (
              <div className="text-center space-y-4 text-neon-red">
                <p className="font-semibold bg-red-950/50 p-4 rounded-lg border border-neon-red/20">{error}</p>
              </div>
            )}

            {result && !loading && (
              <div className="w-full space-y-8 animate-in zoom-in-95 duration-500">
                <div className="text-center space-y-2">
                  <h3 className="text-gray-400 font-medium uppercase tracking-widest text-sm">Click Probability</h3>
                  <div className="text-6xl font-bold text-white tracking-tighter">
                    {(result.click_probability * 100).toFixed(2)}<span className="text-neon-red">%</span>
                  </div>
                </div>

                <div className="space-y-4 pt-6 border-t border-white/10">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 flex items-center"><Zap size={16} className="mr-2 text-neon-red" /> Inference Latency</span>
                    <span className="text-white font-mono">{result.latency_ms} ms</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 flex items-center"><BarChart2 size={16} className="mr-2 text-blue-400" /> Model Version</span>
                    <span className="text-white font-mono bg-blue-900/30 px-2 py-1 rounded text-xs border border-blue-500/30">LGBM-OPT-V1</span>
                  </div>
                </div>
                
                {/* Visual Gauge */}
                <div className="pt-4">
                  <div className="w-full bg-black/60 rounded-full h-3 overflow-hidden border border-white/5">
                    <div 
                      className="bg-gradient-to-r from-orange-500 to-neon-red h-full transition-all duration-1000 ease-out neon-glow"
                      style={{ width: `${Math.min((result.click_probability * 100) * 3, 100)}%` }} // Multiplied by 3 for visual exaggeration of small probabilities
                    ></div>
                  </div>
                  <p className="text-xs text-gray-500 text-center mt-2">Adjusted scale for sparse conversion rates</p>
                </div>
              </div>
            )}

          </div>
        </div>

      </main>
    </div>
  );
}

export default App;
