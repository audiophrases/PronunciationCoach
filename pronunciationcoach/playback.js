(value) => {
    // Gradio's waveform player replaces its media element asynchronously. Wait
    // for the new element instead of relying on its autoplay prop being toggled.
    const request = window.coachPlaybackRequest = (window.coachPlaybackRequest || 0) + 1;
    const status = document.getElementById('playback-message');
    if (status) status.textContent = '';
    if (!value) return [];
    const audioElements = (root) => {
        if (!root) return [];
        return [...root.querySelectorAll('audio'),
            ...[...root.querySelectorAll('*')].flatMap(el =>
                el.shadowRoot ? audioElements(el.shadowRoot) : [])];
    };
    const start = async () => {
        const message = 'Audio could not start automatically. Press Play in “Now playing”. If it is still silent, check the tab mute and browser sound permission.';
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        for (let attempt = 0; attempt < 200; attempt++) {
            if (request !== window.coachPlaybackRequest) return '';
            // There is also an empty recording element inside Gradio's component.
            // WaveSurfer keeps its actual player inside a shadow root.
            const media = audioElements(document.getElementById('player'))
                .find(audio => audio.currentSrc && audio.readyState >= 1);
            if (media) {
                try {
                    media.currentTime = 0;
                    await media.play();
                    return '';
                } catch (error) {
                    // WaveSurfer can pause while installing the new buffer.
                    // Retry that transient interruption, but expose permission
                    // and decoding errors instead of silently swallowing them.
                    if (error.name !== 'AbortError') {
                        console.warn('Pronunciation Coach playback:', error);
                        return message;
                    }
                }
            }
            await new Promise(resolve => setTimeout(resolve, 50));
        }
        return 'Audio did not finish loading. Try the playback button again.';
    };
    // Return immediately so Gradio can finish rendering this value. Awaiting
    // media here can hold up the very component update we are waiting for.
    start().then(message => {
        if (request === window.coachPlaybackRequest && status) status.textContent = message;
    });
    return [];
}
