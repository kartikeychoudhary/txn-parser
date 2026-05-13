// Shared frontend utilities — escape + regex-based JSON syntax highlighter.

export function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}

// Tokenises BEFORE HTML-escaping so the quoted-string regex stays simple.
// Returns an HTML string. Caller can wrap it in <pre><code>...</code></pre>.
export function highlightJson(value) {
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  const re = /"(?:\\.|[^"\\])*"(\s*:)?|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\b(?:true|false|null)\b/g;
  const out = [];
  let i = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > i) out.push(escapeHtml(text.slice(i, m.index)));
    const tok = m[0];
    if (tok.startsWith('"')) {
      const colon = m[1];
      if (colon) {
        const keyText = tok.slice(0, tok.length - colon.length);
        out.push(`<span class="tok-key">${escapeHtml(keyText)}</span>${escapeHtml(colon)}`);
      } else {
        out.push(`<span class="tok-str">${escapeHtml(tok)}</span>`);
      }
    } else if (tok === 'true' || tok === 'false' || tok === 'null') {
      out.push(`<span class="tok-lit">${tok}</span>`);
    } else {
      out.push(`<span class="tok-num">${tok}</span>`);
    }
    i = re.lastIndex;
  }
  if (i < text.length) out.push(escapeHtml(text.slice(i)));
  return out.join('');
}

// Render highlighted JSON as a list of <span class="line">…</span> rows so
// the caller can apply per-line classes (e.g. for diff marking).
export function highlightJsonLines(value) {
  const html = highlightJson(value);
  return html.split('\n').map((line) => `<span class="line">${line || ' '}</span>`).join('\n');
}

// Try several strategies to pull a JSON object out of model output.
// Mirrors scripts/_lib.py::extract_json so what the viewer shows lines up
// with what the eval pipeline counts as "JSON valid".
export function extractJson(text) {
  if (!text) return null;
  const candidates = [text.trim(), stripFence(text), extractBraces(text)];
  for (const cand of candidates) {
    if (!cand) continue;
    try {
      const obj = JSON.parse(cand);
      if (obj && typeof obj === 'object' && !Array.isArray(obj)) return obj;
    } catch { /* try next */ }
  }
  return null;
}

function stripFence(text) {
  const m = text.trim().match(/^```(?:json)?\s*([\s\S]*?)\s*```\s*$/);
  return m ? m[1] : null;
}

function extractBraces(text) {
  let depth = 0;
  let start = -1;
  let inStr = false;
  let esc = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === '{') { if (depth === 0) start = i; depth++; }
    else if (ch === '}') { depth--; if (depth === 0 && start >= 0) return text.slice(start, i + 1); }
  }
  return null;
}
