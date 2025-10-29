import React, { useState } from 'react';
import './AddPatientForm.css';

const AddPatientForm = ({ onAddPatient }) => {
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};
    
    if (!name.trim()) {
      newErrors.name = 'Name is required';
    } else if (name.trim().length < 2) {
      newErrors.name = 'Name must be at least 2 characters';
    }
    
    if (!age) {
      newErrors.age = 'Age is required';
    } else if (age < 1 || age > 120) {
      newErrors.age = 'Age must be between 1 and 120';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!validateForm()) {
      return;
    }
    onAddPatient({ name: name.trim(), age: parseInt(age, 10) });
    setName('');
    setAge('');
    setErrors({});
  };

  return (
    <div className="add-patient-section">
      <h3 className="section-title">Add New Patient</h3>
      <form onSubmit={handleSubmit} className="patient-form">
        <div className="form-row">
          <div className="form-field">
            <input
              type="text"
              placeholder="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className={errors.name ? 'error' : ''}
              maxLength="50"
            />
            {errors.name && <span className="error-text">{errors.name}</span>}
          </div>
          <div className="form-field">
            <input
              type="number"
              placeholder="Age"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className={errors.age ? 'error' : ''}
              min="1"
              max="120"
            />
            {errors.age && <span className="error-text">{errors.age}</span>}
          </div>
          <button type="submit" className="add-btn">Add Patient</button>
        </div>
      </form>
    </div>
  );
};

export default AddPatientForm;
