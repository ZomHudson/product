import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { AlertCircle, TrendingUp, Package, Calendar, Bell, Check, X, RefreshCw } from 'lucide-react';

const ChickenRestockDashboard = () => {
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [weeklyPredictions, setWeeklyPredictions] = useState([]);
  const [historicalData, setHistoricalData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [accuracy, setAccuracy] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Backend API URL - change this to your Flask server URL
  const API_BASE_URL = 'http://localhost:5000/api';

  useEffect(() => {
    fetchAllData();
    // Auto-refresh every 5 minutes
    const interval = setInterval(fetchAllData, 300000);
    return () => clearInterval(interval);
  }, []);

  const fetchAllData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch current prediction
      const predictionRes = await fetch(`${API_BASE_URL}/predict`);
      const predictionData = await predictionRes.json();
      if (predictionData.success) {
        setCurrentPrediction(predictionData.data);
      }

      // Fetch weekly predictions
      const weeklyRes = await fetch(`${API_BASE_URL}/predict/week`);
      const weeklyData = await weeklyRes.json();
      if (weeklyData.success) {
        setWeeklyPredictions(weeklyData.data);
      }

      // Fetch historical data
      const historyRes = await fetch(`${API_BASE_URL}/history?days=30`);
      const historyData = await historyRes.json();
      if (historyData.success) {
        processHistoricalData(historyData.data);
      }

      // Fetch accuracy stats
      const accuracyRes = await fetch(`${API_BASE_URL}/accuracy?days=30`);
      const accuracyData = await accuracyRes.json();
      if (accuracyData.success) {
        setAccuracy(accuracyData.data);
      }

      // Fetch alerts
      const alertsRes = await fetch(`${API_BASE_URL}/alerts`);
      const alertsData = await alertsRes.json();
      if (alertsData.success) {
        setAlerts(alertsData.data.map((alert, idx) => ({ ...alert, id: idx })));
      }

    } catch (err) {
      setError(`Failed to fetch data: ${err.message}`);
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const processHistoricalData = (history) => {
    const chartData = history.map(record => {
      const date = new Date(record.timestamp);
      return {
        date: date.toLocaleDateString('en-MY', { month: 'short', day: 'numeric' }),
        predicted: record.prediction.predicted_quantity,
        actual: record.actual || null,
        stock: record.prediction.current_stock.total,
        price: record.prediction.ex_farm_price
      };
    }).reverse();
    
    setHistoricalData(chartData);
  };

  const recordActualQuantity = async (date, actualQuantity) => {
    try {
      const response = await fetch(`${API_BASE_URL}/record`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date, actual_quantity: actualQuantity })
      });
      
      const data = await response.json();
      if (data.success) {
        alert('Actual quantity recorded successfully!');
        fetchAllData(); // Refresh data
      }
    } catch (err) {
      alert(`Failed to record: ${err.message}`);
    }
  };

  const dismissAlert = (id) => {
    setAlerts(alerts.filter(alert => alert.id !== id));
  };

  if (loading && !currentPrediction) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-orange-50 to-yellow-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-orange-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 font-medium">Loading predictions from API...</p>
        </div>
      </div>
    );
  }

  if (error && !currentPrediction) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-orange-50 to-yellow-50 flex items-center justify-center p-6">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-md">
          <div className="text-red-600 text-center mb-4">
            <AlertCircle className="w-16 h-16 mx-auto mb-4" />
            <h2 className="text-xl font-bold mb-2">Connection Error</h2>
            <p className="text-gray-600 mb-4">{error}</p>
            <p className="text-sm text-gray-500 mb-4">
              Make sure the Flask API server is running on http://localhost:5000
            </p>
            <button
              onClick={fetchAllData}
              className="px-6 py-3 bg-orange-600 text-white rounded-lg font-semibold hover:bg-orange-700 transition-colors"
            >
              Retry Connection
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!currentPrediction) return null;

  const stockPercentage = (currentPrediction.current_stock.total / 2000) * 100;
  const isLowStock = currentPrediction.current_stock.total < 500;
  const isCritical = currentPrediction.current_stock.total < 300;

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-yellow-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8 flex justify-between items-center">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">🐔 Chicken Restock Dashboard</h1>
            <p className="text-gray-600">Real-time predictions powered by Python ML</p>
          </div>
          <button
            onClick={fetchAllData}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-white rounded-lg shadow hover:shadow-lg transition-all disabled:opacity-50"
          >
            <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
            <span className="font-medium">Refresh</span>
          </button>
        </div>

        {/* Connection Status */}
        <div className="mb-6 bg-green-50 border border-green-200 rounded-lg p-3 flex items-center gap-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-green-700 font-medium">Connected to Flask API Server</span>
        </div>

        {/* Alerts Section */}
        {alerts.length > 0 && (
          <div className="mb-6 space-y-3">
            {alerts.map(alert => (
              <div
                key={alert.id}
                className={`p-4 rounded-lg shadow-sm flex items-start justify-between ${
                  alert.type === 'critical' ? 'bg-red-50 border-l-4 border-red-500' :
                  alert.type === 'warning' ? 'bg-yellow-50 border-l-4 border-yellow-500' :
                  'bg-blue-50 border-l-4 border-blue-500'
                }`}
              >
                <div className="flex items-start gap-3 flex-1">
                  <Bell className={`w-5 h-5 mt-0.5 ${
                    alert.type === 'critical' ? 'text-red-600' :
                    alert.type === 'warning' ? 'text-yellow-600' :
                    'text-blue-600'
                  }`} />
                  <div>
                    <h3 className="font-semibold text-gray-900">{alert.message}</h3>
                    <p className="text-sm text-gray-600 mt-1">{alert.detail}</p>
                    <p className="text-xs text-gray-500 mt-2">
                      {new Date(alert.timestamp).toLocaleString('en-MY')}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => dismissAlert(alert.id)}
                  className="text-gray-400 hover:text-gray-600 ml-4"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Main Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-orange-500">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-orange-100 rounded-lg">
                <Package className="w-6 h-6 text-orange-600" />
              </div>
              <span className={`text-xs font-semibold px-3 py-1 rounded-full ${
                isCritical ? 'bg-red-100 text-red-600' :
                isLowStock ? 'bg-yellow-100 text-yellow-600' :
                'bg-green-100 text-green-600'
              }`}>
                {isCritical ? 'Critical' : isLowStock ? 'Low' : 'Good'}
              </span>
            </div>
            <h3 className="text-gray-600 text-sm font-medium mb-1">Current Stock</h3>
            <p className="text-3xl font-bold text-gray-900">{currentPrediction.current_stock.total}</p>
            <div className="mt-3 bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${
                  isCritical ? 'bg-red-500' :
                  isLowStock ? 'bg-yellow-500' :
                  'bg-green-500'
                }`}
                style={{ width: `${Math.min(stockPercentage, 100)}%` }}
              />
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Factory: {currentPrediction.current_stock.factory} | Kiosk: {currentPrediction.current_stock.kiosk}
            </p>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-blue-500">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <TrendingUp className="w-6 h-6 text-blue-600" />
              </div>
              <span className="text-xs font-semibold px-3 py-1 rounded-full bg-blue-100 text-blue-600">
                {currentPrediction.confidence}
              </span>
            </div>
            <h3 className="text-gray-600 text-sm font-medium mb-1">Next Restock</h3>
            <p className="text-3xl font-bold text-gray-900">{currentPrediction.predicted_quantity}</p>
            <p className="text-xs text-gray-500 mt-2">units recommended</p>
            <p className="text-xs text-blue-600 mt-1 font-medium">
              {currentPrediction.factors.total_adjustment}
            </p>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-green-500">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-100 rounded-lg">
                <Calendar className="w-6 h-6 text-green-600" />
              </div>
            </div>
            <h3 className="text-gray-600 text-sm font-medium mb-1">Next Restock Date</h3>
            <p className="text-xl font-bold text-gray-900">{currentPrediction.target_date}</p>
            <div className="mt-3 p-2 bg-green-50 rounded-lg">
              <p className="text-xs text-green-700 font-medium">{currentPrediction.calendar_event.event_name}</p>
              <p className="text-xs text-green-600 mt-1">Impact: {(currentPrediction.calendar_event.factor * 100).toFixed(0)}%</p>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-purple-500">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-purple-100 rounded-lg">
                <span className="text-2xl">💰</span>
              </div>
            </div>
            <h3 className="text-gray-600 text-sm font-medium mb-1">Ex-Farm Price</h3>
            <p className="text-3xl font-bold text-gray-900">RM {currentPrediction.ex_farm_price.toFixed(2)}</p>
            <p className="text-xs text-gray-500 mt-2">per kg</p>
            <p className="text-xs text-purple-600 mt-1 font-medium">
              {currentPrediction.factors.price_adjustment}
            </p>
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Historical Accuracy */}
          {historicalData.length > 0 && (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Prediction Accuracy (30 Days)</h2>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={historicalData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={2} name="Predicted" />
                  {historicalData.some(d => d.actual) && (
                    <Line type="monotone" dataKey="actual" stroke="#10b981" strokeWidth={2} name="Actual" />
                  )}
                </LineChart>
              </ResponsiveContainer>
              {accuracy && (
                <div className="mt-4 grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 rounded-lg p-3">
                    <p className="text-xs text-blue-600 font-medium">Avg Accuracy</p>
                    <p className="text-2xl font-bold text-blue-700">{accuracy.avg_accuracy.toFixed(1)}%</p>
                  </div>
                  <div className="bg-green-50 rounded-lg p-3">
                    <p className="text-xs text-green-600 font-medium">Total Predictions</p>
                    <p className="text-2xl font-bold text-green-700">{accuracy.total_predictions}</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Stock Trends */}
          {historicalData.length > 0 && (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Stock & Price Trends</h2>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={historicalData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="left" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  />
                  <Legend />
                  <Area yAxisId="left" type="monotone" dataKey="stock" fill="#fb923c" stroke="#f97316" name="Stock Level" />
                  <Line yAxisId="right" type="monotone" dataKey="price" stroke="#8b5cf6" strokeWidth={2} name="Price (RM)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Weekly Predictions */}
        {weeklyPredictions.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Upcoming Week Predictions</h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-3 text-sm font-semibold text-gray-700">Date</th>
                    <th className="text-left p-3 text-sm font-semibold text-gray-700">Predicted Qty</th>
                    <th className="text-left p-3 text-sm font-semibold text-gray-700">Event</th>
                    <th className="text-left p-3 text-sm font-semibold text-gray-700">Adjustment</th>
                    <th className="text-left p-3 text-sm font-semibold text-gray-700">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {weeklyPredictions.map((pred, idx) => (
                    <tr key={idx} className="border-b hover:bg-gray-50">
                      <td className="p-3 text-sm">{pred.target_date}</td>
                      <td className="p-3 text-sm font-semibold">{pred.predicted_quantity}</td>
                      <td className="p-3 text-sm">{pred.calendar_event.event_name}</td>
                      <td className="p-3 text-sm">{pred.factors.total_adjustment}</td>
                      <td className="p-3 text-sm">
                        <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                          {pred.confidence}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Adjustment Factors */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Adjustment Factors Breakdown</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {Object.entries(currentPrediction.factors).map(([key, value]) => {
              const isPositive = value.includes('+');
              const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
              return (
                <div key={key} className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-xs text-gray-600 mb-2">{label}</p>
                  <p className={`text-xl font-bold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                    {value}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChickenRestockDashboard;