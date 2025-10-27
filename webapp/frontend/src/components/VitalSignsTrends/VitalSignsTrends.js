import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './VitalSignsTrends.css';

const VitalSignsTrends = ({ data }) => {
  return (
    <div className="vital-signs-trends-card">
      <h3>Vital Signs Trends</h3>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={data}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="systolic_bp" stroke="#8884d8" activeDot={{ r: 8 }} name="Systolic BP" />
            <Line type="monotone" dataKey="diastolic_bp" stroke="#82ca9d" name="Diastolic BP" />
            <Line type="monotone" dataKey="heart_rate" stroke="#ffc658" name="Heart Rate" />
             <Line type="monotone" dataKey="body_temp" stroke="#fc5c7d" name="Body Temp (°C)" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default VitalSignsTrends;
