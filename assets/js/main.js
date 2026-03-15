(() => {
  const toastWrap = document.querySelector('[data-toast-wrap]');

  function showToast(message, type = 'info', timeout = 2800) {
    if (!toastWrap) return;
    const toast = document.createElement('div');
    toast.className = `toast${type === 'error' ? ' is-error' : ''}`;
    toast.textContent = message;
    toastWrap.appendChild(toast);
    window.setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(8px)';
      window.setTimeout(() => toast.remove(), 220);
    }, timeout);
  }

  function initReveal() {
    const nodes = document.querySelectorAll('.reveal, .fade-scale');
    if (!nodes.length) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('show');
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.14 });

    nodes.forEach((node) => io.observe(node));
  }

  function initYear() {
    const yearNode = document.querySelector('[data-year]');
    if (yearNode) yearNode.textContent = String(new Date().getFullYear());
  }

  function initSupport() {
    const node = document.querySelector('[data-support-link]');
    if (!node) return;
    node.addEventListener('click', () => {
      showToast('Открываем поддержку в Telegram.');
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    initReveal();
    initYear();
    initSupport();
    showToast('Сайт Voice PRO AI обновлён.');
  });

  window.VoiceProSite = {
    showToast,
  };
})();
