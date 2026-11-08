(function () {
  const toggle = document.getElementById('theme-toggle');
  const icon = toggle.querySelector('.theme-toggle-icon');
  const root = document.documentElement;

  // Apply stored theme on load
  const stored = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = stored || (prefersDark ? 'dark' : 'light');

  setTheme(theme);

  toggle.addEventListener('click', function () {
    const current = root.getAttribute('data-theme') || 'light';
    setTheme(current === 'dark' ? 'light' : 'dark');
  });

  function setTheme(theme) {
    root.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    if (icon) icon.textContent = theme === 'dark' ? '☀️' : '🌙';
  }
})();
