(value) => {
    const request = window.coachPlaybackRequest = (window.coachPlaybackRequest || 0) + 1;
    const media = document.getElementById('coach-audio');
    const status = document.getElementById('playback-message');
    if (!media) return [];
    if (status) status.textContent = '';
    media.pause();
    if (!value) {
        media.removeAttribute('src');
        media.load();
        return [];
    }
    // Keep one native element alive; waveform clear/restore races its load handlers.
    const url = value.url;
    if (!url) {
        if (status) status.textContent = 'No playable audio was returned. Try again.';
        return [];
    }
    if (media.getAttribute('src') !== url) {
        media.src = url;
        media.load();
    } else {
        media.currentTime = 0;
    }
    const timer = setTimeout(() => {
        if (request === window.coachPlaybackRequest && status) {
            status.textContent = 'Audio is taking too long to load. Try the playback button again.';
        }
    }, 10000);
    media.play().then(() => {
        clearTimeout(timer);
        if (request === window.coachPlaybackRequest && status) status.textContent = '';
    }).catch(error => {
        clearTimeout(timer);
        if (request !== window.coachPlaybackRequest) return;
        console.warn('Pronunciation Coach playback:', error);
        if (status) status.textContent = 'Audio could not start automatically. Press Play in Now playing. If it is still silent, check the tab mute and browser sound permission.';
    });
    // Let Gradio finish its event while the browser decodes the audio.
    return [];
}
