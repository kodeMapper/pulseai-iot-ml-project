import React, { useState } from 'react';
import './AddPatientForm.css';

const AddPatientForm = ({ onAddPatient }) => {
  const [name, setName] = useState('');
  const [age, setAge] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!name || !age) {
      alert('Please enter name and age.');
      return;
    }
    onAddPatient({ name, age: parseInt(age, 10) });
    setName('');
    setAge('');
  };

  return (
    <div className="add-patient-form">
      <h3>Add New Patient</h3>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <input
          type="number"
          placeholder="Age"
          value={age}
          onChange={(e) => setAge(e.target.value)}
        />
        <button type="submit">Add Patient</button>
      </form>
    </div>
  );
};

export default AddPatientForm;
