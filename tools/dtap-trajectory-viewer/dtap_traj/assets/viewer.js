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
      return `<article class="row tool ${event.injection_target ? 'target' : ''}"><div class="icon">🔧</div><div class="body"><div class="label">Tool call${event.injection_target ? ' <span class="tag">injection target</span>' : ''}</div><div class="tool-name"><code>${esc(event.server)}</code><code>${esc(event.tool)}</code> ${args(event.args, event.arg_injection_spans)}</div>${description}</div></article>`;
    }
    const pair = labels[event.kind];
    if (!pair) return '';
    const injected = event.injection_spans && event.injection_spans.length;
    const pre = event.kind === 'tool_result' || event.kind === 'judge';
    return `<article class="row ${esc(event.kind)} ${injected ? 'target' : ''}"><div class="icon">${pair[0]}</div><div class="body"><div class="label">${pair[1]}${injected ? ' <span class="tag">contains submitted payload</span>' : ''}</div>${pre ? '<pre>' : '<div class="text">'}${spans(event.text, event.injection_spans)}${pre ? '</pre>' : '</div>'}</div></article>`;
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
      `<div class="payload"><span class="tag">${esc(p.kind || 'attack')}</span>${p.tool ? ` <code>${esc(p.tool)}</code>` : ''}<pre>${esc(p.text)}</pre></div>`
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
      <details open><summary>Unified diff</summary><pre class="diff">${renderDiff(comparison.diff)}</pre></details>
      <div class="config-grid"><details><summary>Original config.yaml</summary><pre>${esc(comparison.original)}</pre></details><details><summary>Submitted config.yaml</summary><pre>${esc(comparison.submitted)}</pre></details></div>`;
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
  renderComparison();
  renderPayloads();
  renderTimeline('policy-timeline', data.policy_timeline, 'No policy trajectory was supplied. Use --policy-trace.');
  renderTimeline('victim-timeline', data.timeline, 'No DTAP victim trajectory was found.');
  selectPanel((data.policy_timeline || []).length ? 'policy-panel' : 'victim-panel');
})();
