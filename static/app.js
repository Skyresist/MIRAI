// MIRAI kiosk screen: welcome -> chat (voice or text) -> idle/finish -> welcome for the next visitor.
'use strict';

const TEXT = {
  id: {
    finish: 'Selesai', send: 'Kirim', cancel: '✕ Batal', placeholder: 'Ketik pertanyaan…', talk: 'Mulai bertanya', type: 'Ketik',
    stopTalking: 'Selesai bicara', topics: 'Topik populer', close: 'Tutup', typeLabel: 'Ketik pertanyaanmu',
    touchStart: 'Sentuh layar untuk memulai',
    welcome: 'Halo! Aku MIRAI, siap membantu mencari informasi tentang BINUS ASO.',
    stillThere: 'Masih di sana? Sentuh layar untuk melanjutkan.',
    greeting: 'Halo! Aku bisa membantu mencari informasi tentang BINUS ASO. Tekan Mulai Bertanya, ketik, atau pilih Topik Populer.',
    tapToTalk: 'Tekan Mulai Bertanya untuk bertanya', listening: 'Aku mendengarkan… tekan Selesai Bicara kalau sudah',
    transcribing: 'Memproses suara…', thinking: 'Sebentar, aku sedang mencari jawabannya', speaking: 'MIRAI sedang menjawab… tekan tombol oranye untuk memotong',
    noSpeech: 'Maaf, saya tidak mendengar dengan jelas. Coba lagi lebih dekat ke mikrofon.',
    micBlocked: 'Mikrofon tidak tersedia. Silakan ketik pertanyaan Anda.',
    offline: 'MIRAI sedang tidak dapat menjawab. Silakan hubungi staf di sekitar.',
    busy: 'Tunggu sebentar, MIRAI masih menjawab.',
    suggestions: ['Apa itu BINUS ASO?', 'Ada jurusan apa saja?', 'Apakah ada beasiswa?', 'Ada UKM apa saja?',
      'Kampusnya di mana?', 'Bagaimana cara mendaftar?', 'Fasilitasnya apa saja?', 'Harus bisa bahasa Jepang?'],
  },
  en: {
    finish: 'Finish', send: 'Send', cancel: '✕ Cancel', placeholder: 'Type a question…', talk: 'Start asking', type: 'Type',
    stopTalking: 'Done talking', topics: 'Popular topics', close: 'Close', typeLabel: 'Type your question',
    touchStart: 'Touch the screen to start',
    welcome: "Hi! I'm MIRAI, here to help you find information about BINUS ASO.",
    stillThere: 'Still there? Touch the screen to continue.',
    greeting: 'Hi! I can help you find information about BINUS ASO. Tap Start Asking, type, or pick a Popular Topic.',
    tapToTalk: 'Tap Start Asking to ask a question', listening: "I'm listening… tap Done Talking when you finish",
    transcribing: 'Processing your voice…', thinking: 'One moment, I am looking for the answer', speaking: 'MIRAI is answering… tap the orange button to interrupt',
    noSpeech: "Sorry, I didn't catch that. Please try again closer to the microphone.",
    micBlocked: 'The microphone is unavailable. Please type your question.',
    offline: 'MIRAI cannot answer right now. Please ask nearby staff.',
    busy: 'One moment, MIRAI is still answering.',
    suggestions: ['What is BINUS ASO?', 'What majors are offered?', 'Are there scholarships?', 'What student clubs are there?',
      'Where is the campus?', 'How do I register?', 'What facilities are there?', 'Do I need to speak Japanese?'],
  },
};

const $ = (selector) => document.querySelector(selector);
const state = { language: 'id', busy: false, audio: null, idleTimer: null, warnTimer: null, idleSeconds: 120, voice: true, abort: null, undo: null, speaking: null, volume: 1 };
const recorder = new TapToTalkRecorder({
  onLevel: (level) => { $('.mic-level').style.borderWidth = `${Math.round(level * 18)}px`; $('.mic-level').style.inset = `${-Math.round(level * 18)}px`; },
});

function t(key) { return TEXT[state.language][key]; }

// The face shows what MIRAI is doing; a timed mood (happy, confused) falls back to idle.
// It never moves or resizes: only the eyes change.
let faceTimer = null;
function setFace(mood, ms = 0) {
  clearTimeout(faceTimer);
  $('#face').dataset.mood = mood;
  if (ms) faceTimer = setTimeout(() => setFace('idle'), ms);
  updateAnswering();
}

// While MIRAI shows an answer, the answer fills the face box and the eyes move to its top-right
// corner; listening, thinking or a puzzled look bring the full face back.
function updateAnswering() {
  const last = $('#messages').lastElementChild;
  const answer = last && last.matches('.mirai:not(.thinking):not(.greeting)');
  $('#canvas').classList.toggle('answering', Boolean(answer) && ['idle', 'speaking', 'happy'].includes($('#face').dataset.mood));
}

// The staff design draws everything on a 1366 x 769 canvas; scale it to fit whatever screen this is.
function fitCanvas() {
  document.documentElement.style.setProperty('--scale', Math.min(innerWidth / 1366, innerHeight / 769));
}

// The "..." settings and the popular topics open above their buttons; opening one closes the other.
function togglePopup(button, popup, open = popup.hidden) {
  [[$('#more'), $('#menu')], [$('#topics'), $('#suggestions')]].forEach(([b, p]) => {
    p.hidden = !(open && p === popup);
    b.setAttribute('aria-expanded', String(!p.hidden));
  });
}
const closePopups = () => togglePopup($('#topics'), $('#suggestions'), false);

function setVolume(volume) {
  state.volume = volume;
  if (state.audio) state.audio.volume = volume;
  $('#volume').value = Math.round(volume * 100);
  $('#volume-label').textContent = `${Math.round(volume * 100)}%`;
}

// On-screen keyboard: keys write into the text box; Shift capitalises the next letter only.
const KEY_ROWS = [['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'], ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
  ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', "'"], ['shift', 'z', 'x', 'c', 'v', 'b', 'n', 'm', 'back'], [',', 'space', '.', '?']];
let shift = false;

function buildKeyboard() {
  $('#keys').replaceChildren(...KEY_ROWS.map((row) => {
    const line = document.createElement('div');
    line.className = 'key-row';
    line.append(...row.map((key) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `key${key === 'space' ? ' space' : key.length > 1 ? ' wide' : ''}`;
      button.dataset.key = key;
      button.textContent = { shift: '⇧', back: '⌫', space: ' ' }[key] ?? key;
      if (key === 'back') button.setAttribute('aria-label', 'Backspace');
      if (key === 'space') button.setAttribute('aria-label', 'Space');
      if (key === 'shift') button.setAttribute('aria-label', 'Shift');
      return button;
    }));
    return line;
  }));
}

function pressKey(key) {
  const input = $('#text');
  const start = input.selectionStart ?? input.value.length;
  const end = input.selectionEnd ?? start;
  let insert = key === 'space' ? ' ' : key;
  if (key === 'shift') { shift = !shift; return renderShift(); }
  if (key === 'back') {
    const from = start === end ? Math.max(0, start - 1) : start;
    input.value = input.value.slice(0, from) + input.value.slice(end);
    input.setSelectionRange(from, from);
    return;
  }
  if (shift) { insert = insert.toUpperCase(); shift = false; renderShift(); }
  if (input.value.length - (end - start) + insert.length > input.maxLength) return;
  input.value = input.value.slice(0, start) + insert + input.value.slice(end);
  input.setSelectionRange(start + insert.length, start + insert.length);
}

function renderShift() {
  document.querySelectorAll('.key').forEach((button) => {
    const key = button.dataset.key;
    if (key === 'shift') button.classList.toggle('on', shift);
    else if (key.length === 1) button.textContent = shift ? key.toUpperCase() : key;
  });
}

function showKeyboard(open) {
  $('#keyboard').hidden = !open;
  $('#chat').classList.toggle('typing', open);
  $('#type').setAttribute('aria-expanded', String(open));
  if (open) { stopSpeaking(); closePopups(); $('#text').focus(); }
  else { $('#text').value = ''; shift = false; renderShift(); }
}

function setStatus(message = t('tapToTalk'), isError = false) {
  $('#status').textContent = message;
  $('#status').classList.toggle('error', isError);
}

function applyLanguage(language) {
  state.language = language;
  document.documentElement.lang = language;
  document.querySelectorAll('[data-i18n]').forEach((node) => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll('[data-i18n-placeholder]').forEach((node) => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  document.querySelectorAll('.lang button').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.language === language)));
  renderSuggestions();
  if (!state.busy && !recorder.recording) setStatus();
}

function renderSuggestions() {
  const box = $('#suggestions');
  box.replaceChildren(...t('suggestions').map((question) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = question;
    button.addEventListener('click', () => { closePopups(); ask(question); });
    return button;
  }));
}

function addMessage(role, text, { error = false, greeting = false } = {}) {
  const item = document.createElement('li');
  item.className = `message ${role}${error ? ' error' : ''}${greeting ? ' greeting' : ''}`;
  // "[1]" citation markers mean nothing to a visitor, and the kiosk shows no source line.
  item.textContent = text.replace(/\s*\[\d+\]/g, '');
  $('#messages').append(item);
  updateAnswering();
  return item;
}

// The cancel button shows whenever there is something to take back: recording, thinking or speaking.
function updateCancel() {
  $('#cancel').hidden = !($('#mic').classList.contains('listening') || state.busy || state.audio || state.speaking);
  $('#topics').hidden = $('#type').hidden = !$('#cancel').hidden;  // Cancel takes their place meanwhile.
  if (!$('#cancel').hidden) { $('#suggestions').hidden = true; $('#topics').setAttribute('aria-expanded', 'false'); }
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
  if ($('#face').dataset.mood === 'speaking') setFace('idle');
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
  setFace('idle');
  setStatus();
}

// Plays one WAV; resolves when it ends or is paused (cancel, a new question, Talk).
function play(blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.volume = state.volume;
    state.audio = audio;
    updateCancel();
    setStatus(t('speaking'));
    audio.onplaying = () => { if (state.audio === audio) setFace('speaking'); };
    audio.onended = audio.onpause = () => {
      URL.revokeObjectURL(url);
      if (state.audio === audio) state.audio = null;
      resolve();
    };
    audio.play().catch(reject);
  });
}

// Piper synthesises on the CPU, so the whole answer took seconds before any sound. Speaking it
// sentence by sentence starts the voice after the first one, while the next is made during playback.
async function speak(text, language) {
  if (!state.voice) return;
  const token = state.speaking = {};
  updateCancel();
  const sentences = text.replace(/\s*\[\d+\]/g, '').split(/(?<=[.!?])\s+/).filter((sentence) => sentence.trim());
  const synthesize = (sentence) => api('/api/speak', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: sentence, language }),
  }).then((response) => response.blob());
  try {
    let next = synthesize(sentences[0]);
    for (let index = 0; index < sentences.length; index += 1) {
      const blob = await next;
      if (state.speaking !== token) return;  // Cancelled, or a new question came in meanwhile.
      if (index + 1 < sentences.length) {
        next = synthesize(sentences[index + 1]);
        next.catch(() => {});  // Reported when awaited; never as an unhandled rejection after a cancel.
      }
      await play(blob);
    }
  } catch (error) {
    console.warn('Speech unavailable:', error);  // Text already answered the visitor.
  } finally {
    if (state.speaking === token) {
      state.speaking = null;
      if ($('#face').dataset.mood === 'speaking') setFace('idle');
      updateCancel();
      if (!state.busy && !recorder.recording) setStatus();
      resetIdle();
    }
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
  setFace('thinking');
  setStatus(t('thinking'));
  try {
    const response = await api('/api/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: question, language: state.language }), signal: state.abort.signal,
    });
    const answer = await response.json();
    state.abort = state.undo = null;
    pending.remove();
    addMessage('mirai', answer.text);
    setBusy(false);
    setFace('idle');
    setStatus();
    speak(answer.text, answer.language);
  } catch (error) {
    if (error.name === 'AbortError') return;  // cancel() already cleaned up.
    state.abort = state.undo = null;
    pending.remove();
    addMessage('mirai', error.status === 409 ? t('busy') : t('offline'), { error: true });
    setBusy(false);
    setFace('confused', 2500);
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
    showKeyboard(false);
    $('#mic').classList.add('listening');
    $('#mic-label').textContent = t('stopTalking');
    setFace('listening');
    setStatus(t('listening'));
    updateCancel();
    result = await recorder.record();
  } catch (error) {
    console.warn('Microphone error:', error);
    setFace('confused', 2500);
    setStatus(t('micBlocked'), true);
    return;
  } finally {
    $('#mic').classList.remove('listening');
    $('#mic-label').textContent = t('talk');
    updateCancel();
  }
  if (!result) { setFace('idle'); return setStatus(); }
  if (!result.heardSpeech) { setFace('confused', 2500); return setStatus(t('noSpeech'), true); }
  state.abort = new AbortController();
  setBusy(true);
  setFace('thinking');
  setStatus(t('transcribing'));
  try {
    const form = new FormData();
    form.append('audio', result.blob, result.blob.type.includes('ogg') ? 'speech.ogg' : 'speech.webm');
    form.append('language', state.language);
    const transcript = await (await api('/api/transcribe', { method: 'POST', body: form, signal: state.abort.signal })).json();
    state.abort = null;
    setBusy(false);
    if (!transcript.text) { setFace('confused', 2500); return setStatus(t('noSpeech'), true); }
    await ask(transcript.text);
  } catch (error) {
    if (error.name === 'AbortError') return;
    state.abort = null;
    setBusy(false);
    setFace('confused', 2500);
    setStatus(error.status === 503 ? t('micBlocked') : t('offline'), true);
  }
}

// One visitor at a time: finishing or walking away clears memory and returns to the welcome screen.
async function endVisit() {
  clearTimeout(state.idleTimer); clearTimeout(state.warnTimer);
  recorder.cancel();
  stopSpeaking();
  showKeyboard(false);
  setFace('sleeping');
  closePopups();
  applyLanguage('id');  // The next visitor starts in Indonesian; the menu switches.
  $('#idle-warning').hidden = true;
  $('#messages').replaceChildren();
  $('#chat').hidden = true;
  $('#welcome').hidden = false;
  $('#canvas').classList.add('welcome-visible');
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
  closePopups();
  $('#welcome').hidden = true;
  $('#canvas').classList.remove('welcome-visible');
  $('#chat').hidden = false;
  addMessage('mirai', t('greeting'), { greeting: true });
  setFace('happy', 2000);
  resetIdle();
  if (state.voice) recorder.open().catch(() => setStatus(t('micBlocked'), true));
}

// Touching anywhere on the sleeping screen wakes MIRAI (the "..." menu is not part of it).
$('#welcome').addEventListener('click', () => startVisit(state.language));
$('#more').addEventListener('click', () => { resetIdle(); togglePopup($('#more'), $('#menu')); });
$('#volume').addEventListener('input', () => { setVolume($('#volume').value / 100); resetIdle(); });
$('#topics').addEventListener('click', () => { resetIdle(); togglePopup($('#topics'), $('#suggestions')); });
document.querySelectorAll('.lang button').forEach((button) => button.addEventListener('click', () => { applyLanguage(button.dataset.language); resetIdle(); }));
// A tap outside an open popup closes it.
document.addEventListener('pointerdown', (event) => {
  if (!event.target.closest('.popup, #more, #topics')) closePopups();
});
addEventListener('resize', fitCanvas);
fitCanvas();
$('#finish').addEventListener('click', () => endVisit());
$('#mic').addEventListener('click', onMic);
$('#cancel').addEventListener('click', cancel);
$('#text-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const question = $('#text').value;
  if (!question.trim() || state.busy) return;
  showKeyboard(false);
  ask(question);
});
$('#type').addEventListener('click', () => { resetIdle(); showKeyboard($('#keyboard').hidden); });
// Tapping the face while it is free makes MIRAI smile.
$('#face').addEventListener('click', () => {
  if ($('#face').dataset.mood === 'idle') setFace('happy', 1500);
});
$('#close-keyboard').addEventListener('click', () => showKeyboard(false));
// pointerdown + preventDefault: tapping a key must not take focus (and the caret) away from the text box.
$('#keys').addEventListener('pointerdown', (event) => {
  const key = event.target.closest('.key');
  if (!key) return;
  event.preventDefault();
  pressKey(key.dataset.key);
});
buildKeyboard();
['pointerdown', 'keydown'].forEach((name) => document.addEventListener(name, () => { if (!$('#chat').hidden) resetIdle(); }, { passive: true }));

api('/api/status').then((response) => response.json()).then((status) => {
  state.idleSeconds = status.idle_timeout || state.idleSeconds;
  state.voice = status.voice !== false;
  $('#mic').hidden = !state.voice;
}).catch(() => {});
applyLanguage('id');
setFace('sleeping');
$('#canvas').classList.add('welcome-visible');
