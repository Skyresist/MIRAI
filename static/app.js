// MIRAI kiosk screen: welcome -> chat (voice or text) -> idle/finish -> welcome for the next visitor.
'use strict';

const TEXT = {
  id: {
    finish: 'Selesai', send: 'Kirim', cancel: '✕ Batal', placeholder: 'Ketik pertanyaan…',
    stillThere: 'Masih di sana? Sentuh layar untuk melanjutkan.',
    greeting: 'Halo! Saya MIRAI. Silakan tekan tombol mikrofon lalu bicara, atau pilih pertanyaan di bawah.',
    tapToTalk: 'Tekan mikrofon untuk bicara', listening: 'Mendengarkan… (tekan lagi untuk selesai)',
    transcribing: 'Memproses suara…', thinking: 'MIRAI sedang berpikir', speaking: 'MIRAI berbicara… (tekan mikrofon untuk memotong)',
    noSpeech: 'Maaf, saya tidak mendengar dengan jelas. Coba lagi lebih dekat ke mikrofon.',
    micBlocked: 'Mikrofon tidak tersedia. Silakan ketik pertanyaan Anda.',
    offline: 'MIRAI sedang tidak dapat menjawab. Silakan hubungi staf di sekitar.',
    busy: 'Tunggu sebentar, MIRAI masih menjawab.', source: 'Sumber',
    suggestions: ['Apa itu BINUS ASO?', 'Ada jurusan apa saja?', 'Apakah ada beasiswa?', 'Ada UKM apa saja?',
      'Kampusnya di mana?', 'Bagaimana cara mendaftar?', 'Fasilitasnya apa saja?', 'Harus bisa bahasa Jepang?'],
  },
  en: {
    finish: 'Finish', send: 'Send', cancel: '✕ Cancel', placeholder: 'Type a question…',
    stillThere: 'Still there? Touch the screen to continue.',
    greeting: "Hi! I'm MIRAI. Tap the microphone and speak, or choose a question below.",
    tapToTalk: 'Tap the microphone to talk', listening: 'Listening… (tap again when done)',
    transcribing: 'Processing your voice…', thinking: 'MIRAI is thinking', speaking: 'MIRAI is speaking… (tap the mic to interrupt)',
    noSpeech: "Sorry, I didn't catch that. Please try again closer to the microphone.",
    micBlocked: 'The microphone is unavailable. Please type your question.',
    offline: 'MIRAI cannot answer right now. Please ask nearby staff.',
    busy: 'One moment, MIRAI is still answering.', source: 'Source',
    suggestions: ['What is BINUS ASO?', 'What majors are offered?', 'Are there scholarships?', 'What student clubs are there?',
      'Where is the campus?', 'How do I register?', 'What facilities are there?', 'Do I need to speak Japanese?'],
  },
};

const $ = (selector) => document.querySelector(selector);
const state = { language: 'id', busy: false, audio: null, idleTimer: null, warnTimer: null, idleSeconds: 120, voice: true, abort: null, undo: null, speaking: null };
const recorder = new TapToTalkRecorder({
  onLevel: (level) => { $('.mic-level').style.borderWidth = `${Math.round(level * 18)}px`; $('.mic-level').style.inset = `${-Math.round(level * 18)}px`; },
});

function t(key) { return TEXT[state.language][key]; }

function setStatus(message = t('tapToTalk'), isError = false) {
  $('#status').textContent = message;
  $('#status').classList.toggle('error', isError);
}

function applyLanguage(language) {
  state.language = language;
  document.documentElement.lang = language;
  document.querySelectorAll('[data-i18n]').forEach((node) => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll('[data-i18n-placeholder]').forEach((node) => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  document.querySelectorAll('.segmented button').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.language === language)));
  renderSuggestions();
  if (!state.busy && !recorder.recording) setStatus();
}

function renderSuggestions() {
  const box = $('#suggestions');
  box.replaceChildren(...t('suggestions').map((question) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = question;
    button.addEventListener('click', () => ask(question));
    return button;
  }));
}

function addMessage(role, text, { sources = [], error = false } = {}) {
  const item = document.createElement('li');
  item.className = `message ${role}${error ? ' error' : ''}`;
  // "[1]" markers mean nothing to a visitor; the Source line below already says where it came from.
  item.textContent = text.replace(/\s*\[\d+\]/g, '');
  if (sources.length) {
    const note = document.createElement('div');
    note.className = 'sources';
    note.textContent = `${t('source')}: ` + sources.map((source) => source.question || `${source.source} p. ${source.page}`).join(' · ');
    item.append(note);
  }
  $('#messages').append(item);
  item.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return item;
}

// The cancel button shows whenever there is something to take back: recording, thinking or speaking.
function updateCancel() {
  $('#cancel').hidden = !($('#mic').classList.contains('listening') || state.busy || state.audio || state.speaking);
}

function setBusy(busy) {
  state.busy = busy;
  updateCancel();
  $('#mic').classList.toggle('busy', busy);
  $('#text-form button').disabled = busy;
  $('#suggestions').querySelectorAll('button').forEach((button) => { button.disabled = busy; });
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const error = new Error(detail.detail || response.statusText);
    error.status = response.status;
    throw error;
  }
  return response;
}

function stopSpeaking() {
  state.speaking = null;  // An answer still being synthesised must not start playing later.
  if (state.audio) { state.audio.pause(); state.audio = null; }
  updateCancel();
}

// One tap undoes the current turn: stop listening, drop the question being answered, or stop speech.
function cancel() {
  resetIdle();
  recorder.cancel();
  stopSpeaking();
  if (state.abort) {
    api('/api/cancel', { method: 'POST' }).catch(() => {});  // The server drops the answer from memory.
    state.abort.abort();
    state.abort = null;
    state.undo?.();
    state.undo = null;
    setBusy(false);
  }
  setStatus();
}

async function speak(text, language) {
  if (!state.voice) return;
  const token = state.speaking = {};
  updateCancel();
  try {
    const response = await api('/api/speak', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, language }),
    });
    const blob = await response.blob();
    if (state.speaking !== token) return;  // Cancelled, or a new question came in meanwhile.
    const url = URL.createObjectURL(blob);
    stopSpeaking();
    const audio = new Audio(url);
    state.audio = audio;
    updateCancel();
    setStatus(t('speaking'));
    audio.onended = audio.onpause = () => {
      URL.revokeObjectURL(url);
      if (state.audio === audio) state.audio = null;
      updateCancel();
      if (!state.busy && !recorder.recording) setStatus();
      resetIdle();
    };
    await audio.play();
  } catch (error) {
    console.warn('Speech unavailable:', error);  // Text already answered the visitor.
    if (state.speaking === token) { state.speaking = null; updateCancel(); setStatus(); }
  }
}

async function ask(question) {
  question = question.trim();
  if (!question || state.busy) return;
  stopSpeaking();
  resetIdle();
  const asked = addMessage('visitor', question);
  const pending = addMessage('mirai thinking', t('thinking'));
  pending.append(Object.assign(document.createElement('span'), { className: 'dots' }));
  state.abort = new AbortController();
  state.undo = () => { asked.remove(); pending.remove(); };
  setBusy(true);
  setStatus(t('thinking'));
  try {
    const response = await api('/api/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: question, language: state.language }), signal: state.abort.signal,
    });
    const answer = await response.json();
    state.abort = state.undo = null;
    pending.remove();
    addMessage('mirai', answer.text, { sources: answer.sources });
    setBusy(false);
    setStatus();
    speak(answer.text, answer.language);
  } catch (error) {
    if (error.name === 'AbortError') return;  // cancel() already cleaned up.
    state.abort = state.undo = null;
    pending.remove();
    addMessage('mirai', error.status === 409 ? t('busy') : t('offline'), { error: true });
    setBusy(false);
    // Visitors see the friendly message; the small status line tells staff what actually failed.
    if (error.status === 409) setStatus();
    else setStatus(`(${error.status || 'no connection'}) ${error.message}`.slice(0, 160), true);
    console.error('Ask failed:', error);
  } finally {
    resetIdle();
  }
}

async function onMic() {
  resetIdle();
  if (recorder.recording) return recorder.stop();
  if (state.busy) return;
  stopSpeaking();
  let result;
  try {
    $('#mic').classList.add('listening');
    setStatus(t('listening'));
    updateCancel();
    result = await recorder.record();
  } catch (error) {
    console.warn('Microphone error:', error);
    setStatus(t('micBlocked'), true);
    return;
  } finally {
    $('#mic').classList.remove('listening');
    updateCancel();
  }
  if (!result) return setStatus();
  if (!result.heardSpeech) return setStatus(t('noSpeech'), true);
  state.abort = new AbortController();
  setBusy(true);
  setStatus(t('transcribing'));
  try {
    const form = new FormData();
    form.append('audio', result.blob, result.blob.type.includes('ogg') ? 'speech.ogg' : 'speech.webm');
    form.append('language', state.language);
    const transcript = await (await api('/api/transcribe', { method: 'POST', body: form, signal: state.abort.signal })).json();
    state.abort = null;
    setBusy(false);
    if (!transcript.text) return setStatus(t('noSpeech'), true);
    await ask(transcript.text);
  } catch (error) {
    if (error.name === 'AbortError') return;
    state.abort = null;
    setBusy(false);
    setStatus(error.status === 503 ? t('micBlocked') : t('offline'), true);
  }
}

// One visitor at a time: finishing or walking away clears memory and returns to the welcome screen.
async function endVisit() {
  clearTimeout(state.idleTimer); clearTimeout(state.warnTimer);
  recorder.cancel();
  stopSpeaking();
  $('#idle-warning').hidden = true;
  $('#messages').replaceChildren();
  $('#chat').hidden = true;
  $('#welcome').hidden = false;
  try { await api('/api/reset', { method: 'POST' }); } catch (error) { console.warn('Reset failed:', error); }
}

function resetIdle() {
  clearTimeout(state.idleTimer); clearTimeout(state.warnTimer);
  $('#idle-warning').hidden = true;
  if ($('#chat').hidden) return;
  const warnAt = Math.max(5, state.idleSeconds - 15);
  state.warnTimer = setTimeout(() => {
    if (state.busy || state.audio || recorder.recording) return resetIdle();
    $('#idle-warning').hidden = false;
  }, warnAt * 1000);
  state.idleTimer = setTimeout(() => {
    if (state.busy || state.audio || recorder.recording) return resetIdle();
    endVisit();
  }, state.idleSeconds * 1000);
}

function startVisit(language) {
  applyLanguage(language);
  $('#welcome').hidden = true;
  $('#chat').hidden = false;
  addMessage('mirai', t('greeting'));
  resetIdle();
  if (state.voice) recorder.open().catch(() => setStatus(t('micBlocked'), true));
}

document.querySelectorAll('.big-choice').forEach((button) => button.addEventListener('click', () => startVisit(button.dataset.language)));
document.querySelectorAll('.segmented button').forEach((button) => button.addEventListener('click', () => { applyLanguage(button.dataset.language); resetIdle(); }));
$('#finish').addEventListener('click', endVisit);
$('#mic').addEventListener('click', onMic);
$('#cancel').addEventListener('click', cancel);
$('#text-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const question = $('#text').value;
  $('#text').value = '';
  ask(question);
});
['pointerdown', 'keydown'].forEach((name) => document.addEventListener(name, () => { if (!$('#chat').hidden) resetIdle(); }, { passive: true }));

api('/api/status').then((response) => response.json()).then((status) => {
  state.idleSeconds = status.idle_timeout || state.idleSeconds;
  state.voice = status.voice !== false;
  $('#mic').hidden = !state.voice;
}).catch(() => {});
applyLanguage('id');
