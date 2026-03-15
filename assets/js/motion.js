(() => {
  function initAmbientMotion() {
    const layer = document.querySelector('[data-ambient]');
    if (!layer) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    let rafId = 0;
    let x = 0;
    let y = 0;

    function apply() {
      layer.style.transform = `translate3d(${x * 10}px, ${y * 10}px, 0)`;
      rafId = 0;
    }

    window.addEventListener('pointermove', (event) => {
      const w = window.innerWidth || 1;
      const h = window.innerHeight || 1;
      x = (event.clientX / w - 0.5) * 1.4;
      y = (event.clientY / h - 0.5) * 1.2;
      if (!rafId) rafId = window.requestAnimationFrame(apply);
    }, { passive: true });
  }

  document.addEventListener('DOMContentLoaded', initAmbientMotion);
})();
