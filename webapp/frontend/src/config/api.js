// API configuration for production/development
const API_BASE_URL = process.env.REACT_APP_API_URL || '';

export const getApiUrl = (path) => {
  // In development, proxy handles the routing (empty base)
  // In production, use the full backend URL
  return `${API_BASE_URL}${path}`;
};

export default API_BASE_URL;
