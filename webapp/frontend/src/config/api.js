// API configuration for production/development
// In production (Vercel), use the Render backend URL
// In development, the proxy in package.json handles routing

const PRODUCTION_API_URL = 'https://pulseai-backend-7xlo.onrender.com';

// Use environment variable if set, otherwise use production URL in production build
const API_BASE_URL = process.env.REACT_APP_API_URL || 
  (process.env.NODE_ENV === 'production' ? PRODUCTION_API_URL : '');

export const getApiUrl = (path) => {
  return `${API_BASE_URL}${path}`;
};

export default API_BASE_URL;
