import React from 'react';
import './PatientHeader.css';

const PatientHeader = ({ patient }) => {
  if (!patient) return null;

  const getInitials = (name) => {
    if (!name) return '';
    const names = name.split(' ');
    if (names.length > 1) {
      return `${names[0][0]}${names[1][0]}`;
    }
    return names[0][0];
  };

  return (
    <div className="patient-header-card">
      <div className="patient-avatar">
        <span>{getInitials(patient.name)}</span>
      </div>
      <div className="patient-info">
        <h1>{patient.name}</h1>
        <p>
          <span>ID: {patient.patient_id}</span>
          <span>&bull;</span>
          <span>{patient.age} years</span>
          <span>&bull;</span>
          <span>{patient.gender}</span>
          <span>&bull;</span>
          <span>{patient.contact}</span>
        </p>
      </div>
    </div>
  );
};

export default PatientHeader;
