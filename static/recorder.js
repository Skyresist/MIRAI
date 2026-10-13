// Tap-to-talk recorder: records from the default microphone (the Jetson's USB mic in kiosk mode)
// and stops by itself after the visitor finishes speaking.
'use strict';

class TapToTalkRecorder {
  constructor({ onLevel = () => {}, silenceMs = 1200, noSpeechMs = 7000, maxMs = 15000 } = {}) {
    Object.assign(this, { onLevel, silenceMs, noSpeechMs, maxMs });
    this.stream = null;
    this.active = null;
  }

  async open() {
    if (this.stream) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.context = new AudioContext();
    const source = this.context.createMediaStreamSource(this.stream);
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 1024;
    source.connect(this.analyser);
  }

  close() {
    this.cancel();
    this.stream?.getTracks().forEach((track) => track.stop());
    this.context?.close();
    this.stream = this.context = this.analyser = null;
  }

  level() {
    const samples = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(samples);
    let sum = 0;
    for (const sample of samples) sum += sample * sample;
    return Math.sqrt(sum / samples.length);
  }

  // Resolves with { blob, heardSpeech }. Call stop() to finish early (second tap).
  async record() {
    await this.open();
    if (this.context.state === 'suspended') await this.context.resume();
    const mimeType = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm']
      .find((type) => MediaRecorder.isTypeSupported(type)) || '';
    const recorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : {});
    const chunks = [];
    recorder.ondataavailable = (event) => event.data.size && chunks.push(event.data);

    return new Promise((resolve, reject) => {
      const started = performance.now();
      let noiseFloor = 0, calibrationFrames = 0, heardSpeech = false, lastSpeech = started, cancelled = false;
      const finish = () => recorder.state !== 'inactive' && recorder.stop();
      recorder.onerror = (event) => reject(event.error || new Error('Recording failed'));
      recorder.onstop = () => {
        cancelAnimationFrame(this.frame);
        this.active = null;
        this.onLevel(0);
        if (cancelled) return resolve(null);
        resolve({ blob: new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }), heardSpeech });
      };

      const tick = () => {
        const now = performance.now(), rms = this.level();
        if (now - started < 300) {
          noiseFloor = (noiseFloor * calibrationFrames + rms) / ++calibrationFrames;
        } else {
          // Speech must stand clearly above this room's own background noise.
          const threshold = Math.max(0.015, noiseFloor * 2.5);
          if (rms > threshold) { heardSpeech = true; lastSpeech = now; }
          this.onLevel(Math.min(1, rms / (threshold * 3)));
          if ((heardSpeech && now - lastSpeech > this.silenceMs)
              || (!heardSpeech && now - started > this.noSpeechMs)
              || now - started > this.maxMs) return finish();
        }
        this.frame = requestAnimationFrame(tick);
      };

      this.active = { stop: finish, cancel: () => { cancelled = true; finish(); } };
      recorder.start(250);
      tick();
    });
  }

  stop() { this.active?.stop(); }
  cancel() { this.active?.cancel(); }
  get recording() { return this.active !== null; }
}

window.TapToTalkRecorder = TapToTalkRecorder;
