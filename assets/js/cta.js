(() => {
  const DOWNLOAD_URL = 'https://github.com/dencikpapinka-png/ai-voice-pro-ai/releases/latest/download/Voice_PRO_AI_Setup.exe';
  const MANIFEST_URLS = [
    '/update.json',
    'https://raw.githubusercontent.com/dencikpapinka-png/ai-voice-pro-ai/gh-pages/update.json',
  ];

  function safeText(value) {
    return String(value ?? '').trim();
  }

  function formatPublishedAt(isoValue) {
    const raw = safeText(isoValue);
    if (!raw) return '—';
    const dt = new Date(raw);
    if (Number.isNaN(dt.getTime())) return raw;
    return new Intl.DateTimeFormat('ru-RU', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(dt);
  }

  function shortHash(hash) {
    const value = safeText(hash).toLowerCase();
    if (!value) return '—';
    if (value.length <= 24) return value;
    return `${value.slice(0, 12)}…${value.slice(-12)}`;
  }

  async function loadManifest() {
    for (const url of MANIFEST_URLS) {
      try {
        const res = await fetch(url, { cache: 'no-store' });
        if (!res.ok) continue;
        const data = await res.json();
        if (data && data.stable) return data.stable;
      } catch (_) {
        // try next source
      }
    }
    return null;
  }

  async function initReleaseProof() {
    const versionNode = document.querySelector('[data-release-version]');
    const hashNode = document.querySelector('[data-release-sha]');
    const dateNode = document.querySelector('[data-release-published]');
    if (!versionNode || !hashNode || !dateNode) return;

    try {
      const stable = await loadManifest();
      if (!stable) throw new Error('manifest_not_found');
      const version = safeText(stable.version);
      const sha = safeText(stable.sha256);
      const publishedAt = safeText(stable.published_at);
      versionNode.textContent = version ? `v${version}` : '—';
      hashNode.textContent = shortHash(sha);
      hashNode.title = sha || '';
      dateNode.textContent = formatPublishedAt(publishedAt);
    } catch (_) {
      versionNode.textContent = 'недоступно';
      hashNode.textContent = 'недоступно';
      dateNode.textContent = 'недоступно';
    }
  }

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

  document.addEventListener('DOMContentLoaded', () => {
    initDownloadLinks();
    initReleaseProof();
  });
})();
