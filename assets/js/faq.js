(() => {
  function toggleFaq(button) {
    const expanded = button.getAttribute('aria-expanded') === 'true';
    const panelId = button.getAttribute('aria-controls');
    const panel = document.getElementById(panelId);
    if (!panel) return;

    button.setAttribute('aria-expanded', String(!expanded));
    panel.hidden = expanded;

    const icon = button.querySelector('.faq-icon');
    if (icon) {
      icon.style.transform = expanded ? 'rotate(0deg)' : 'rotate(180deg)';
    }
  }

  function initFaq() {
    const buttons = document.querySelectorAll('[data-faq-button]');
    buttons.forEach((button) => {
      button.addEventListener('click', () => toggleFaq(button));
      button.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          toggleFaq(button);
        }
      });
    });
  }

  document.addEventListener('DOMContentLoaded', initFaq);
})();
