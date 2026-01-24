import { Routes, Route, Navigate } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import CheckoutPage from './pages/CheckoutPage';
import { getToken } from './api/client';

export default function App() {
  const isAuthed = !!getToken();

  return (
    <Routes>
      {/* Default route -> checkout (will redirect to login if not authed) */}
      <Route path="/" element={<Navigate to="/checkout" replace />} />

      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      <Route
        path="/checkout"
        element={isAuthed ? <CheckoutPage /> : <Navigate to="/login" replace />}
      />

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/checkout" replace />} />
    </Routes>
  );
}