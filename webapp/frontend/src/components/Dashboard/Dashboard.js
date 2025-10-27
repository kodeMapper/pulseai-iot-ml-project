import React, { useState, useEffect } from 'react';
import Summary from '../Summary/Summary';
import PatientDirectory from '../PatientDirectory/PatientDirectory';
import AddPatientForm from '../AddPatientForm/AddPatientForm';
import './Dashboard.css';

const Dashboard = () => {
  const [patients, setPatients] = useState([]);

  const fetchPatients = async () => {
    const response = await fetch('/api/patients');
    const data = await response.json();
    setPatients(data);
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  const handleAddPatient = async (patient) => {
    const response = await fetch('/api/patients', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(patient),
    });
    if (response.ok) {
      fetchPatients(); // Refresh the patient list
    } else {
      alert('Failed to add patient.');
    }
  };

  return (
    <div className="dashboard">
      <Summary patients={patients} />
      <AddPatientForm onAddPatient={handleAddPatient} />
      <PatientDirectory patients={patients} />
    </div>
  );
};

export default Dashboard;
