import React from 'react';
import { Link } from 'react-router-dom';
import './PatientDirectory.css';

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
              <td>{patient.last_check_in ? new Date(patient.last_check_in.$date).toLocaleDateString() : 'N/A'}</td>
              <td>
                <Link to={`/patients/${patient._id}`}>
                  <button>Details</button>
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
