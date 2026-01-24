import { useState, useEffect } from 'react';
import './styles/base.css';
import Login from './pages/Login';
import Decisions from './pages/Decisions';
import Config from './pages/Config';
import Rules from './pages/Rules';
import Coupons from './pages/Coupons';

export default function App() {
  const [authed, setAuthed] = useState(!!localStorage.getItem('token'));

  // current page: 'decisions' | 'config' | 'rules'
  const [page, setPage] = useState('decisions');

  // initialize page from a global or fallback
  useEffect(() => {
    // if something already set window._page earlier, respect it
    const init = window._page || 'decisions';
    setPage(init);
    window._page = init;
  }, []);

  const handleNav = (nextPage) => (e) => {
    e.preventDefault();
    window._page = nextPage;
    setPage(nextPage); // trigger React re-render
  };

  if (!authed) {
    return <Login onAuthed={() => setAuthed(true)} />;
  }

  return (
    <div>
      <div className="container">
        <div className="nav">
          <a
            className={`button${page === 'decisions' ? '' : ' ghost'}`}
            href="#"
            onClick={handleNav('decisions')}
          >
            Decisions
          </a>
          <a
            className={`button${page === 'config' ? '' : ' ghost'}`}
            href="#"
            onClick={handleNav('config')}
          >
            Config
          </a>
          <a
            className={`button${page === 'rules' ? '' : ' ghost'}`}
            href="#"
            onClick={handleNav('rules')}
          >
            Rules
          </a>
          <a
              className={`button${page === 'coupons' ? '' : ' ghost'}`}
              href="#"
              onClick={handleNav('coupons')}
            >
              Coupons
          </a>
          <a
            className="button ghost"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              localStorage.removeItem('token');
              window.location.reload();
            }}
          >
            Logout
          </a>
        </div>
      </div>
      {page === 'config' ? (
        <Config />
      ) : page === 'rules' ? (
        <Rules />
      ) : page === 'coupons' ? (
        <Coupons />
      ) : (
        <Decisions />
      )}
    </div>
  );
}
