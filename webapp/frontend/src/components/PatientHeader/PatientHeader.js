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

  const renderPatientDetails = () => {
    const details = [];
    
    if (patient.patient_id) details.push(`ID: ${patient.patient_id}`);
    if (patient.age) details.push(`${patient.age} years`);
    if (patient.gender) details.push(patient.gender);
    if (patient.contact) details.push(patient.contact);

    return details.map((detail, index) => (
      <React.Fragment key={index}>
        <span>{detail}</span>
        {index < details.length - 1 && <span>&bull;</span>}
      </React.Fragment>
    ));
  };

  return (
    <div className="patient-header-card">
      <div className="patient-avatar">
        <span>{getInitials(patient.name)}</span>
      </div>
      <div className="patient-info">
        <h1>{patient.name}</h1>
        <p>
          {renderPatientDetails()}
        </p>
      </div>
    </div>
  );
};

export default PatientHeader;
