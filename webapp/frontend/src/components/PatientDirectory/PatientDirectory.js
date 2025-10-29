import React from 'react';
import { Link } from 'react-router-dom';
import './PatientDirectory.css';

const normalizeTimestamp = (value) => {
  if (!value) return null;
  if (typeof value === 'string') {
    return new Date(value);
  }
  if (typeof value === 'object' && value.$date) {
    return new Date(value.$date);
  }
  return null;
};

const PatientDirectory = ({ patients }) => {
  return (
    <div className="patient-directory">
      <h2>Patient Directory</h2>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Age</th>
            <th>Risk Level</th>
            <th>Last Check-in</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {patients.map((patient) => (
            <tr key={patient._id}>
              <td>{patient.name}</td>
              <td>{patient.age}</td>
              <td>{patient.risk_level || 'N/A'}</td>
              <td>{(() => {
                const parsedDate = normalizeTimestamp(patient.last_check_in);
                return parsedDate ? parsedDate.toLocaleDateString() : 'N/A';
              })()}</td>
              <td>
                <Link to={`/patients/${patient._id}`} className="details-link">
                  <button className="details-btn">
                    <span className="details-icon">👤</span>
                    <span>View Details</span>
                  </button>
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default PatientDirectory;
