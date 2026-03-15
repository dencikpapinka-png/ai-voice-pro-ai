(() => {
  function initNavMenu() {
    const toggle = document.querySelector('[data-menu-toggle]');
    const nav = document.querySelector('[data-nav-links]');
    if (!toggle || !nav) return;

    toggle.addEventListener('click', () => {
      const open = nav.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', String(open));
    });

    nav.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        if (window.innerWidth <= 900) {
          nav.classList.remove('is-open');
          toggle.setAttribute('aria-expanded', 'false');
        }
      });
    });
  }

  function initActiveLinks() {
    const sections = Array.from(document.querySelectorAll('section[id]'));
    const links = Array.from(document.querySelectorAll('[data-nav-links] a[href^="#"]'));
    if (!sections.length || !links.length) return;

    const map = new Map(links.map((link) => [link.getAttribute('href').slice(1), link]));

    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const id = entry.target.id;
        links.forEach((link) => link.classList.remove('is-active'));
        const active = map.get(id);
        if (active) active.classList.add('is-active');
      });
    }, { rootMargin: '-30% 0px -55% 0px', threshold: 0.01 });

    sections.forEach((section) => io.observe(section));
  }

  document.addEventListener('DOMContentLoaded', () => {
    initNavMenu();
    initActiveLinks();
  });
})();
