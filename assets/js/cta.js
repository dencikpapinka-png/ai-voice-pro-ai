(() => {
  const DOWNLOAD_URL = 'https://github.com/dencikpapinka-png/ai-voice-pro-ai/releases/latest/download/Voice_PRO_AI_Setup.exe';

  function initDownloadLinks() {
    const links = document.querySelectorAll('[data-download-link]');
    links.forEach((link) => {
      link.setAttribute('href', DOWNLOAD_URL);
      link.setAttribute('rel', 'noopener noreferrer');
      link.addEventListener('click', () => {
        if (window.VoiceProSite?.showToast) {
          window.VoiceProSite.showToast('Начинаем загрузку установщика Voice PRO AI...');
        }
      });
    });
  }

  document.addEventListener('DOMContentLoaded', initDownloadLinks);
})();
