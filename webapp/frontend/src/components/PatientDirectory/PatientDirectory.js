import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { StaggerContainer, FadeUpItem, GlowCard, TextReveal } from '../common/MotionWrappers';
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
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('name'); // 'name', 'risk', 'age'

  const filteredAndSortedPatients = useMemo(() => {
    if (!patients) return [];

    let result = [...patients];

    // Filter
    if (searchTerm && searchTerm.trim() !== '') {
      const lowerTerm = searchTerm.toLowerCase().trim();
      result = result.filter(p => 
        p.name.toLowerCase().includes(lowerTerm) || 
        (p.patient_id && p.patient_id.toString().includes(lowerTerm))
      );
    }

    // Sort
    result.sort((a, b) => {
      if (sortBy === 'name') return a.name.localeCompare(b.name);
      if (sortBy === 'age') return a.age - b.age;
      if (sortBy === 'risk') {
        const riskOrder = { 'High': 3, 'Medium': 2, 'Low': 1, 'N/A': 0 };
        return (riskOrder[b.risk_level] || 0) - (riskOrder[a.risk_level] || 0);
      }
      return 0;
    });

    return result;
  }, [patients, searchTerm, sortBy]);

  const clearSearch = () => setSearchTerm('');

  return (
    <div className="patient-directory">
      <div className="directory-header">
        <h2 className="creative-title">
          <span className="title-word">Patient</span>
          <span className="title-word highlight">Directory</span>
        </h2>
        
        <div className="directory-controls">
          <div className="search-bar">
            <span className="search-icon">🔍</span>
            <input 
              type="text" 
              placeholder="Search patients..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            {searchTerm && (
              <button className="clear-search-btn" onClick={clearSearch}>
                ✕
              </button>
            )}
          </div>
          
          <div className="sort-dropdown">
            <span className="sort-label">Sort by:</span>
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              <option value="name">Name</option>
              <option value="risk">Risk Level</option>
              <option value="age">Age</option>
            </select>
          </div>
        </div>
      </div>

      {(!filteredAndSortedPatients || filteredAndSortedPatients.length === 0) ? (
        <div className="empty-state">
          <p>No patients found matching your criteria.</p>
        </div>
      ) : (
        <StaggerContainer className="patient-grid">
          {filteredAndSortedPatients.map((patient) => (
            <FadeUpItem key={patient._id}>
              <GlowCard 
                className="patient-card"
                onClick={() => navigate(`/patients/${patient._id}`)}
              >
                <div className="patient-card-header">
                  <div className="patient-avatar-small">
                    {patient.name.charAt(0)}
                  </div>
                  <div className="patient-risk-badge" data-risk={patient.risk_level}>
                    {patient.risk_level || 'Unknown'}
                  </div>
                </div>
                
                <h3 className="patient-name">{patient.name}</h3>
                
                <div className="patient-stats">
                  <div className="stat">
                    <span className="label">Age</span>
                    <span className="value">{patient.age}</span>
                  </div>
                  <div className="stat">
                    <span className="label">Last Check-in</span>
                    <span className="value">
                      {(() => {
                        const parsedDate = normalizeTimestamp(patient.last_check_in);
                        return parsedDate ? parsedDate.toLocaleDateString() : 'N/A';
                      })()}
                    </span>
                  </div>
                </div>
                
                <div className="card-action">
                  <span>View Profile &rarr;</span>
                </div>
              </GlowCard>
            </FadeUpItem>
          ))}
        </StaggerContainer>
      )}
    </div>
  );
};

export default PatientDirectory;
