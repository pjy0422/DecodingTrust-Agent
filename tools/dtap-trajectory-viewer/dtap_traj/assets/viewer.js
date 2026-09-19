/* Shared renderer for policy and victim timelines. */
(function () {
  'use strict';
  const data = window.DTAP_DATA || {};

  function esc(value) {
    if (value == null) return '';
    return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function spans(text, marks) {
    text = String(text || '');
    if (!marks || !marks.length) return esc(text);
    let result = '', cursor = 0;
    for (const mark of marks.slice().sort((a, b) => a.start - b.start)) {
      if (mark.start < cursor) continue;
      result += esc(text.slice(cursor, mark.start));
      result += `<mark class="inj-mark">${esc(text.slice(mark.start, mark.end))}</mark>`;
      cursor = mark.end;
    }
    return result + esc(text.slice(cursor));
  }

  function args(value, marked) {
    if (!value || typeof value !== 'object') return '<span class="muted">()</span>';
    return Object.entries(value).map(([key, item]) => {
      const raw = typeof item === 'string' ? item : JSON.stringify(item);
      const hit = marked ? marked[key] : null;
      return `<span class="arg"><b>${esc(key)}</b>=${hit ? spans(raw, hit) : esc(raw)}</span>`;
    }).join(' ');
  }

  function eventHtml(event) {
    const labels = {
      user: ['🧑', 'User request'], thinking: ['🧠', 'Agent thinking'],
      tool_result: ['📦', 'Tool result'], say: ['💬', 'Agent says'],
      final: ['✅', 'Final reply'], judge: ['⚖️', 'Judge result']
    };
    if (event.kind === 'tool_call') {
      const desc = event.injected_tool_desc;
      const description = desc ? `<details><summary>Tool description seen by agent</summary><pre>${spans(desc.text, desc.spans)}</pre></details>` : '';
      return `<article class="row tool copy-block ${event.injection_target ? 'target' : ''}"><div class="icon">🔧</div><div class="body"><div class="label">Tool call${event.injection_target ? ' <span class="tag">injection target</span>' : ''}<button type="button" class="copy-button" data-copy-block>Copy</button></div><div data-copy-content><div class="tool-name"><code>${esc(event.server)}</code><code>${esc(event.tool)}</code> ${args(event.args, event.arg_injection_spans)}</div>${description}</div></div></article>`;
    }
    const pair = labels[event.kind];
    if (!pair) return '';
    const injected = event.injection_spans && event.injection_spans.length;
    const pre = event.kind === 'tool_result' || event.kind === 'judge';
    return `<article class="row copy-block ${esc(event.kind)} ${injected ? 'target' : ''}"><div class="icon">${pair[0]}</div><div class="body"><div class="label">${pair[1]}${injected ? ' <span class="tag">contains submitted payload</span>' : ''}<button type="button" class="copy-button" data-copy-block>Copy</button></div>${pre ? '<pre data-copy-content>' : '<div class="text" data-copy-content>'}${spans(event.text, event.injection_spans)}${pre ? '</pre>' : '</div>'}</div></article>`;
  }

  function renderTimeline(id, timeline, emptyText) {
    const box = document.getElementById(id);
    box.classList.remove('loading');
    box.innerHTML = timeline && timeline.length
      ? `<div class="timeline">${timeline.map(eventHtml).join('')}</div>`
      : `<div class="empty">${esc(emptyText)}</div>`;
  }

  function renderPayloads() {
    const box = document.getElementById('payload-panel');
    const payloads = data.payloads || [];
    if (!payloads.length) return;
    box.hidden = false;
    box.innerHTML = `<h2>Submitted attack payloads (${payloads.length})</h2>` + payloads.map((p) =>
      `<div class="payload copy-block"><span class="tag">${esc(p.kind || 'attack')}</span>${p.tool ? ` <code>${esc(p.tool)}</code>` : ''}<button type="button" class="copy-button" data-copy-block>Copy</button><pre data-copy-content>${esc(p.text)}</pre></div>`
    ).join('');
  }

  function renderDiff(diff) {
    if (!diff) return 'No textual differences.';
    return diff.split('\n').map((line) => {
      let cls = 'diff-context';
      if (line.startsWith('@@')) cls = 'diff-hunk';
      else if (line.startsWith('+')) cls = 'diff-add';
      else if (line.startsWith('-')) cls = 'diff-remove';
      return `<span class="${cls}">${esc(line)}</span>`;
    }).join('\n');
  }

  function renderComparison() {
    const comparison = data.config_comparison;
    if (!comparison) return;
    const box = document.getElementById('config-panel');
    box.hidden = false;
    const status = comparison.identical ? 'identical' : 'changed';
    box.innerHTML = `<h2>Configuration comparison <span class="config-status ${status}">${status}</span></h2>
      <div class="paths"><code>${esc(comparison.original_path)}</code> → <code>${esc(comparison.submitted_path)}</code></div>
      <details open class="copy-block"><summary>Unified diff <button type="button" class="copy-button" data-copy-block>Copy</button></summary><pre class="diff" data-copy-content>${renderDiff(comparison.diff)}</pre></details>
      <div class="config-grid"><details class="copy-block"><summary>Original config.yaml <button type="button" class="copy-button" data-copy-block>Copy</button></summary><pre data-copy-content>${esc(comparison.original)}</pre></details><details class="copy-block"><summary>Submitted config.yaml <button type="button" class="copy-button" data-copy-block>Copy</button></summary><pre data-copy-content>${esc(comparison.submitted)}</pre></details></div>`;
  }

  function selectPanel(target) {
    document.querySelectorAll('.trace-panel').forEach((node) => { node.hidden = node.id !== target; });
    document.querySelectorAll('.tabs button').forEach((button) => {
      const selected = button.dataset.target === target;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-selected', String(selected));
    });
  }

  document.querySelectorAll('.tabs button').forEach((button) => button.addEventListener('click', () => selectPanel(button.dataset.target)));
  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const input = document.createElement('textarea');
    input.value = text;
    input.setAttribute('readonly', '');
    input.style.position = 'fixed';
    input.style.opacity = '0';
    document.body.appendChild(input);
    input.select();
    const copied = document.execCommand('copy');
    input.remove();
    if (!copied) throw new Error('clipboard unavailable');
  }
  document.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-copy-block]');
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    const content = button.closest('.copy-block')?.querySelector('[data-copy-content]');
    if (!content) return;
    const original = button.textContent;
    try {
      await copyText(content.innerText || content.textContent || '');
      button.textContent = 'Copied';
    } catch (_) {
      button.textContent = 'Copy failed';
    }
    window.setTimeout(() => { button.textContent = original; }, 1400);
  });
  renderComparison();
  renderPayloads();
  renderTimeline('policy-timeline', data.policy_timeline, 'No policy trajectory was supplied. Use --policy-trace.');
  renderTimeline('victim-timeline', data.timeline, 'No DTAP victim trajectory was found.');
  selectPanel((data.policy_timeline || []).length ? 'policy-panel' : 'victim-panel');
})();
