async function j(path, body) {
  const r = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

export const api = {
  session: (sid) => j('/api/session', { sid }),
  strain: (sid, strain) => j('/api/strain', { sid, strain }),
  chat: (sid, model, prompt) => j('/api/chat', { sid, model, prompt }),
  twin: (sid) => j('/api/twin', { sid }),
  scan: (sid) => j('/api/scan', { sid }),
  operate: (sid, neurons) => j('/api/operate', { sid, neurons }),
  rehab: (sid) => j('/api/rehab', { sid }),
  vitals: (sid) => j('/api/vitals', { sid }),
  reset: (sid) => j('/api/reset', { sid }),
  stage: (sid, stage) => j('/api/stage', { sid, stage }),
  events: (sid) => j('/api/events', { sid }),
  board: () => j('/api/leaderboard'),
};

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
