const MODES = new Set(['trajectories','performance']);
function modeFromLocation(){
  const requested=new URLSearchParams(window.location.search).get('view');
  return MODES.has(requested)?requested:'trajectories';
}
const state = {
  mode: modeFromLocation(),
  facets: null, episodes: [], total: 0, selected: null, tab: 'policy',
  attempt: null,
  page: 0, limit: 100, filters: {run_name:'', domain:'', threat_model:'', status:'', attack_success:'', attack_evaluated:'', q:''},
  cache: new Map(), loading: false, asr: null,
  tuning: {
    facets: null, trials: [], total: 0, selected: null, detail: new Map(),
    compare: new Set(), comparison: null, recommendation: null,
    page: 0, limit: 100, loading: false,
    filters: {phase:'', status:'', hardware_fingerprint:'', model_fingerprint:'', workload_fingerprint:'', software_fingerprint:'', q:''},
  },
};
const $ = (s, root=document) => root.querySelector(s);
const esc = (v='') => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
function qsFilters() {
  const p = new URLSearchParams({limit: state.limit, offset: state.page * state.limit});
  Object.entries(state.filters).forEach(([k,v]) => { if (v !== '' && v !== null) p.set(k,v); });
  return p.toString();
}
async function loadFacets(){ state.facets = await api('/api/facets'); }
async function loadTuningFacets(){ state.tuning.facets = await api('/api/tuning/facets'); if(state.mode==='performance')render(); }
async function loadEpisodes(){
  state.loading = true; render();
  const data = await api(`/api/episodes?${qsFilters()}`);
  state.episodes = data.items; state.total = data.total; state.asr = data.asr; state.loading = false;
  if (!state.selected || !state.episodes.some(x => x.episode_id === state.selected.episode_id)) {
    state.selected = state.episodes[0] || null;
    state.attempt = null;
  }
  render(); if (state.mode==='trajectories'&&state.selected) loadDetail();
}
async function loadDetail(){
  const ep = state.selected; if (!ep) return;
  if(!$('#viewer'))return;
  const suffix=state.attempt===null?'':`?attempt=${state.attempt}`;
  const join=state.attempt===null?'?':'&';
  const key = `${ep.episode_id}@${state.attempt??'latest'}`;
  if (state.cache.has(key)) { renderDetail(); return; }
  $('#viewer').innerHTML = '<div class="loading">Loading trajectory…</div>';
  const [traj, config, judges] = await Promise.all([
    api(`/api/episodes/${encodeURIComponent(ep.episode_id)}/trajectory${suffix}${join}view=combined`),
    api(`/api/episodes/${encodeURIComponent(ep.episode_id)}/config${suffix}`),
    api(`/api/episodes/${encodeURIComponent(ep.episode_id)}/judges${suffix}`),
  ]);
  state.cache.set(key,{...traj, config: config.comparison, judges: judges.judges}); renderDetail();
}
function tuningQuery(includePaging=true){
  const tuning=state.tuning;
  const params=new URLSearchParams();
  if(includePaging){params.set('limit',tuning.limit);params.set('offset',tuning.page*tuning.limit);}
  Object.entries(tuning.filters).forEach(([key,value])=>{if(value!=='')params.set(key,value);});
  return params.toString();
}
async function loadTuningTrials(){
  const tuning=state.tuning; tuning.loading=true; render();
  const [data,recommendation]=await Promise.all([
    api(`/api/tuning/trials?${tuningQuery()}`),
    api(`/api/tuning/recommendations?${tuningQuery(false)}`),
  ]);
  tuning.trials=data.items; tuning.total=data.total; tuning.recommendation=recommendation; tuning.loading=false;
  if(!tuning.selected||!tuning.trials.some(item=>item.trial_id===tuning.selected.trial_id)){
    tuning.selected=tuning.trials[0]||null;
  }
  render(); if(tuning.selected)loadTuningDetail();
}
async function loadTuningDetail(){
  const tuning=state.tuning; const trial=tuning.selected; if(!trial)return;
  if(!tuning.detail.has(trial.trial_id)){
    $('#tuningViewer').innerHTML='<div class="loading">Loading performance trial…</div>';
    tuning.detail.set(trial.trial_id,await api(`/api/tuning/trials/${encodeURIComponent(trial.trial_id)}`));
  }
  renderTuningDetail();
}
async function loadComparison(){
  const ids=[...state.tuning.compare];
  if(ids.length<2){state.tuning.comparison=null;renderTuningDetail();return;}
  const params=new URLSearchParams();ids.forEach(id=>params.append('trial_id',id));
  state.tuning.comparison=await api(`/api/tuning/compare?${params}`);renderTuningDetail();
}
function options(items, selected, all='All') {
  return `<option value="">${all}</option>${(items||[]).map(x=>`<option value="${esc(x.value)}" ${String(x.value)===String(selected)?'selected':''}>${esc(x.value)} · ${x.count}</option>`).join('')}`;
}
function sidebar(){
  const f=state.facets||{}; const s=state.filters;
  return `<aside class="sidebar">
    ${asrPanel()}
    <div class="eyebrow">Run</div><div class="filter"><select data-filter="run_name">${options(f.runs,s.run_name)}</select></div>
    <div class="eyebrow">Trajectory</div>
    <div class="filter"><label>Domain</label><select data-filter="domain">${options(f.domains,s.domain)}</select></div>
    <div class="filter"><label>Threat model</label><select data-filter="threat_model">${options(f.threat_models,s.threat_model)}</select></div>
    <div class="filter"><label>Status</label><select data-filter="status">${options(f.statuses,s.status)}</select></div>
    <div class="filter"><label>Attack judge result</label><div class="toggle attack-toggle">
      <button data-attack="" class="${s.attack_success===''&&s.attack_evaluated===''?'active':''}">All</button>
      <button data-attack="true" class="${s.attack_success==='true'?'active':''}">Succeeded</button>
      <button data-attack="false" class="${s.attack_success==='false'?'active':''}">Failed</button>
      <button data-attack="null" class="${s.attack_evaluated==='false'?'active':''}">Not evaluated</button>
    </div></div>
    <button class="reset" id="resetFilters">Reset filters</button>
  </aside>`;
}
function asrCard(label, value, title){
  const ready=value&&value.rate!==null;
  const percent=ready?`${(value.rate*100).toFixed(1)}%`:'—';
  const ratio=ready?`${value.successes}/${value.evaluated}`:'0/0';
  return `<div class="asr-card" title="${esc(title)}"><span>${esc(label)}</span><b>${percent}</b><small>${ratio}</small></div>`;
}
function asrPanel(){
  const a=state.asr||{};
  return `<div class="eyebrow">Attack success rate</div><div class="asr-grid">
    ${asrCard('H=1',a.h1,'First-submission ASR')}
    ${asrCard('H=2',a.h2,'Second-submission ASR among episodes that reached H=2')}
    ${asrCard('H=1,2',a.cumulative,'Cumulative ASR: either submission succeeded')}
  </div>`;
}
function attackLabel(value, compact=false){
  if(value===true) return [compact?'succeeded':'attack succeeded','bad'];
  if(value===false) return [compact?'failed':'attack failed','ok'];
  return [compact?'not evaluated':'attack not evaluated','neutral'];
}
function count(value){ return value===null||value===undefined?'—':Number(value).toLocaleString(); }
function taskLabel(ep){ return ep.dataset_path||ep.task_id||ep.episode_id; }
function tokenMetrics(actor, usage){
  if(!usage) return `<span class="metric">${actor} tokens <b>not recorded</b></span>`;
  const approximate=usage.reasoning_source==='stream_estimate';
  const without=usage.tokens_without_reasoning===null||usage.tokens_without_reasoning===undefined
    ? 'unavailable':`${approximate?'≈':''}${count(usage.tokens_without_reasoning)}`;
  const thinking=usage.reasoning_tokens===null||usage.reasoning_tokens===undefined
    ? 'unavailable':`${approximate?'≈':''}${count(usage.reasoning_tokens)}`;
  return `<span class="metric token-metric" title="input ${count(usage.input_tokens)}, output ${count(usage.output_tokens)}; thinking ${esc(usage.reasoning_source||'unavailable')}">${actor} tokens incl. thinking <b>${count(usage.tokens_with_reasoning)}</b></span><span class="metric token-metric">${actor} tokens excl. thinking <b>${without}</b></span><span class="metric token-metric">${actor} thinking <b>${thinking}</b></span>`;
}
function theme(){ return document.documentElement.dataset.theme==='light'?'light':'dark'; }
function setTheme(value){
  document.documentElement.dataset.theme=value;
  try { localStorage.setItem('dtap-explorer-theme',value); } catch (_) {}
  const button=$('#themeToggle');
  if(button){
    const light=value==='light';
    button.innerHTML=light?'☾ Dark':'☀ Light';
    button.setAttribute('aria-label',`Switch to ${light?'dark':'light'} theme`);
    button.setAttribute('aria-pressed',String(light));
  }
}
function setMode(value,{replace=false}={}){
  if(!MODES.has(value))return;
  state.mode=value;
  const url=new URL(window.location.href);
  if(value==='trajectories')url.searchParams.delete('view');else url.searchParams.set('view',value);
  window.history[replace?'replaceState':'pushState']({view:value},'',url);
  render();
  if(value==='performance'&&!state.tuning.facets){
    Promise.all([loadTuningFacets(),loadTuningTrials()]).catch(showFailure);
  }else if(value==='performance'&&state.tuning.selected){loadTuningDetail();}
  else if(value==='trajectories'&&state.selected){loadDetail();}
}
function episodeRow(ep){
  const active=state.selected?.episode_id===ep.episode_id;
  const [attackText,attackClass]=attackLabel(ep.attack_success);
  return `<div class="episode ${active?'active':''}" data-episode="${esc(ep.episode_id)}">
    <div class="ep-top"><span class="domain">${esc(ep.domain||'unknown')}</span><span class="threat">${esc(ep.threat_model||'—')}</span></div>
    <div class="ep-id" title="${esc(taskLabel(ep))}">${esc(taskLabel(ep))}</div>
    <div class="ep-run" title="${esc(ep.episode_id)}">${esc(ep.run_name||'run')} · episode ${esc(ep.episode_id)}</div>
    <div class="ep-meta"><span class="${attackClass}"><i class="dot"></i>${attackText}</span><span>${count(ep.policy_events)} policy tool calls</span><span>${count(ep.victim_events)} victim steps</span></div>
  </div>`;
}
function listPane(){
  const start=state.total? state.page*state.limit+1:0, end=Math.min((state.page+1)*state.limit,state.total);
  return `<section class="listpane"><div class="listhead"><div class="listhead-row"><h2>Episodes</h2><span>${start}–${end} / ${state.total}</span></div></div>
    <div class="episodes">${state.loading?'<div class="loading">Index query…</div>':state.episodes.map(episodeRow).join('')}</div>
    <div class="pager"><button id="prev" ${state.page===0?'disabled':''}>← Previous</button><button id="next" ${end>=state.total?'disabled':''}>Next →</button></div></section>`;
}
function detailShell(){
  const ep=state.selected;
  if(!ep) return `<main class="detail"><div class="empty"><div><strong>No trajectory selected</strong>Adjust filters or index a run.</div></div></main>`;
  const [attackText,attackClass]=attackLabel(ep.attack_success,true);
  return `<main class="detail"><div class="detailhead">
      <div class="detail-title"><h1>${esc(taskLabel(ep))}</h1><span class="badge">${esc(ep.domain||'unknown')}</span><span class="badge">${esc(ep.threat_model||'unknown')}</span><span class="badge">${esc(ep.status||ep.episode_status||'unknown')}</span></div>
      <div class="episode-ref">run ${esc(ep.run_name||'unknown')} · config task ${esc(ep.task_id||'not recorded')} · internal episode ${esc(ep.episode_id)}</div>
      <div class="metrics"><span class="metric">policy model <b>${esc(ep.policy_model||'not recorded')}</b></span><span class="metric">victim model <b>${esc(ep.victim_model||'not recorded')}</b></span><span class="metric">policy tool calls <b>${count(ep.policy_events)}</b></span><span class="metric">victim steps <b>${count(ep.victim_events)}</b></span><span class="metric">placements verified <b>${count(ep.placements_verified??0)}</b></span><span class="metric">attack judge <b class="${attackClass}">${attackText}</b></span>${tokenMetrics('policy',ep.policy_usage)}<span class="victim-token-summary">${tokenMetrics('victim',ep.victim_usage)}</span></div>
    </div><div class="tabs">${['policy','victim','combined','judges','config'].map(t=>`<button data-tab="${t}" class="${state.tab===t?'active':''}">${t==='config'?'Config Diff':t==='judges'?'DTAP Judges':t[0].toUpperCase()+t.slice(1)}</button>`).join('')}</div><div class="viewer" id="viewer"></div></main>`;
}
function eventCard(e,i){
  const kind=e.kind||'event'; const tool=[e.server,e.tool].filter(Boolean).join(':');
  let body=e.text;
  if(kind==='tool_call') body=JSON.stringify(e.args||{},null,2);
  if(!body) body='';
  return `<article class="event"><div class="event-head"><span class="kind ${esc(kind)}">${esc(kind.replace('_',' '))}</span>${tool?`<span class="toolname">${esc(tool)}</span>`:''}<span style="margin-left:auto;color:#657184;font-size:9px">#${i+1}</span></div><div class="event-body">${esc(body)}</div></article>`;
}
function timeline(events){ return `<div class="timeline">${(events||[]).map(eventCard).join('')||'<div class="empty">No events</div>'}</div>`; }
function diffHtml(diff){
  if(!diff) return '<div class="empty">No original/submitted config pair found.</div>';
  return `<pre class="diff">${esc(diff).split('\n').map(line=>{
    const c=line.startsWith('+++')||line.startsWith('---')||line.startsWith('@@')?'hdr':line.startsWith('+')?'add':line.startsWith('-')?'del':'';
    return `<span class="${c}">${line}</span>`;
  }).join('\n')}</pre>`;
}
function submittedConfigHtml(data){
  const yaml=data?.config?.submitted;
  if(!yaml) return '<div class="empty">No submitted config found for this attempt.</div>';
  return `<pre class="submitted-yaml">${esc(yaml)}</pre>`;
}
function judgeOutcome(component){
  if(component.success === null || component.success === undefined) return ['unknown','neutral'];
  if(component.name === 'attack') return attackLabel(component.success);
  return [component.success?'task succeeded':'task failed',component.success?'ok':'bad'];
}
function judgeCard(component){
  const [outcome,outcomeClass]=judgeOutcome(component);
  const source=component.source==='llm_as_judge'?'LLM-as-judge':'Deterministic';
  const metadata=component.metadata&&Object.keys(component.metadata).length
    ? `<details><summary>Judge metadata</summary><pre>${esc(JSON.stringify(component.metadata,null,2))}</pre></details>`:'';
  return `<article class="judge-card"><div class="judge-head"><h3>${esc(component.name)} judge</h3><span class="judge-source ${esc(component.source)}">${source}</span><span class="judge-outcome ${outcomeClass}">${esc(outcome)}</span></div>${component.message?`<div class="judge-message">${esc(component.message)}</div>`:''}${metadata}</article>`;
}
function judgesHtml(judges){
  if(!judges||!judges.available) return '<div class="empty">No DTAP judge artifacts found.</div>';
  const cards=(judges.components||[]).map(judgeCard).join('');
  const firewall=judges.reward_firewall&&Object.keys(judges.reward_firewall).length
    ? `<article class="judge-card firewall"><div class="judge-head"><h3>Reward firewall verdict</h3><span class="judge-source trusted">Trusted projection</span></div><pre>${esc(JSON.stringify(judges.reward_firewall,null,2))}</pre></article>`:'';
  const error=judges.error?`<article class="judge-card error"><div class="judge-head"><h3>Judge error</h3></div><div class="judge-message">${esc(judges.error)}</div></article>`:'';
  return `<div class="judges">${cards}${firewall}${error}</div>`;
}
function detailData(){
  if(!state.selected) return null;
  return state.cache.get(`${state.selected.episode_id}@${state.attempt??'latest'}`);
}
function attemptBar(data){
  const attempts=data?.attempts||[];
  if(attempts.length<2) return '';
  const selected=state.attempt??data.attempt_index;
  return `<div class="attemptbar"><label>Submission attempt</label><select id="attemptSelect">${attempts.map(item=>{
    const verdict=item.attack_success===true?' · attack succeeded':item.attack_success===false?' · attack failed':' · not evaluated';
    return `<option value="${item.index}" ${item.index===selected?'selected':''}>H=${item.index}${verdict}</option>`;
  }).join('')}</select><span>${attempts.length} accepted submissions retained</span></div>`;
}
function renderDetail(){
  const viewer=$('#viewer'); if(!viewer||!state.selected) return;
  const data=detailData();
  if(!data){ viewer.innerHTML='<div class="loading">Loading trajectory…</div>'; return; }
  const victimTokens=$('.victim-token-summary');
  if(victimTokens) victimTokens.innerHTML=tokenMetrics('victim',data.victim_usage);
  let body;
  if(state.tab==='policy') body=timeline(data.policy);
  else if(state.tab==='victim') body=timeline(data.victim);
  else if(state.tab==='combined') body=`<div class="lanes"><section><div class="lane-title">Submitted config · H=${data.attempt_index??1}</div>${submittedConfigHtml(data)}</section><section><div class="lane-title">Victim trajectory · ${(data.victim||[]).length}</div>${timeline(data.victim)}</section></div>`;
  else if(state.tab==='judges') body=judgesHtml(data.judges);
  else body=diffHtml(data.config?.diff);
  viewer.innerHTML=attemptBar(data)+body;
  $('#attemptSelect')?.addEventListener('change',e=>{state.attempt=Number(e.target.value);loadDetail();});
}
function tuningOptions(items,selected,all='All'){
  return `<option value="">${all}</option>${(items||[]).map(item=>`<option value="${esc(item.value)}" ${String(item.value)===String(selected)?'selected':''}>${esc(item.label||item.value)} · ${item.count}</option>`).join('')}`;
}
function tuningSidebar(){
  const tuning=state.tuning;const facets=tuning.facets||{};const filters=tuning.filters;
  const recommendation=tuning.recommendation||{};
  const best=recommendation.recommended;
  const pareto=(recommendation.pareto||[]).slice(0,3);
  return `<aside class="sidebar tuning-sidebar">
    <div class="eyebrow">Best known in cohort</div>
    <div class="recommendation ${best?'ready':''}">${best?`<b>${esc(best.trial_id)}</b><span>${esc(best.objective_name)} · ${formatNumber(best.objective_value)}</span>${pareto.length?`<small>Pareto · ${pareto.map(item=>esc(item.trial_id)).join(' · ')}</small>`:''}`:`<span>${esc(recommendation.reason||'Select a comparable measured cohort.')}</span>`}</div>
    <div class="eyebrow">Performance trials</div>
    <div class="filter"><label>Phase</label><select data-tuning-filter="phase">${tuningOptions(facets.phases,filters.phase)}</select></div>
    <div class="filter"><label>Status</label><select data-tuning-filter="status">${tuningOptions(facets.statuses,filters.status)}</select></div>
    <div class="filter"><label>Hardware</label><select data-tuning-filter="hardware_fingerprint">${tuningOptions(facets.hardware,filters.hardware_fingerprint)}</select></div>
    <div class="filter"><label>Model</label><select data-tuning-filter="model_fingerprint">${tuningOptions(facets.models,filters.model_fingerprint)}</select></div>
    <div class="filter"><label>Workload</label><select data-tuning-filter="workload_fingerprint">${tuningOptions(facets.workloads,filters.workload_fingerprint)}</select></div>
    <button class="reset" id="resetTuningFilters">Reset filters</button>
  </aside>`;
}
function formatNumber(value,digits=2){
  return value===null||value===undefined?'—':Number(value).toLocaleString(undefined,{maximumFractionDigits:digits});
}
function tuningStatusClass(status){return status==='success'?'ok':['planned','running'].includes(status)?'neutral':'bad';}
function tuningRow(trial){
  const tuning=state.tuning;const active=tuning.selected?.trial_id===trial.trial_id;
  const checked=tuning.compare.has(trial.trial_id);const compareDisabled=!checked&&tuning.compare.size>=8;
  return `<div class="episode tuning-row ${active?'active':''}" data-trial="${esc(trial.trial_id)}">
    <div class="ep-top"><span class="domain">${esc(trial.phase)}</span><span class="threat ${tuningStatusClass(trial.status)}">${esc(trial.status)}</span><label class="compare-check"><input type="checkbox" data-compare="${esc(trial.trial_id)}" ${checked?'checked':''} ${compareDisabled?'disabled':''}/> compare</label></div>
    <div class="ep-id" title="${esc(trial.trial_id)}">${esc(trial.trial_id)}</div>
    <div class="ep-run">${esc(trial.hardware_label)} · ${esc(trial.model_label)}</div>
    <div class="ep-meta"><span>${esc(trial.objective_name||'unmeasured')} <b>${formatNumber(trial.objective_value)}</b></span><span>VRAM ${formatNumber(trial.peak_vram_gb)} GB</span></div>
  </div>`;
}
function tuningListPane(){
  const tuning=state.tuning;const start=tuning.total?tuning.page*tuning.limit+1:0;const end=Math.min((tuning.page+1)*tuning.limit,tuning.total);
  return `<section class="listpane"><div class="listhead"><div class="listhead-row"><h2>Tuning trials</h2><span>${start}–${end} / ${tuning.total}</span></div><button id="compareTrials" class="compare-button" ${tuning.compare.size<2?'disabled':''}>Compare ${tuning.compare.size}</button></div>
    <div class="episodes">${tuning.loading?'<div class="loading">Index query…</div>':tuning.trials.map(tuningRow).join('')||'<div class="empty">No tuning trials indexed</div>'}</div>
    <div class="pager"><button id="tuningPrev" ${tuning.page===0?'disabled':''}>← Previous</button><button id="tuningNext" ${end>=tuning.total?'disabled':''}>Next →</button></div></section>`;
}
function tuningDetailShell(){
  const trial=state.tuning.selected;
  if(!trial)return `<main class="detail"><div class="empty"><div><strong>No tuning trial selected</strong>Create or index a manual preflight trial.</div></div></main>`;
  return `<main class="detail"><div class="detailhead"><div class="detail-title"><h1>${esc(trial.trial_id)}</h1><span class="badge">${esc(trial.phase)}</span><span class="badge ${tuningStatusClass(trial.status)}">${esc(trial.status)}</span></div><div class="episode-ref">${esc(trial.hardware_label)} · ${esc(trial.model_label)} · ${esc(trial.workload_label)}</div><div class="metrics">${performanceMetrics(trial)}</div></div><div class="tabs"><button class="active">Performance / Tuning</button></div><div class="viewer" id="tuningViewer"></div></main>`;
}
function performanceMetrics(trial){
  const wait=trial.wait_ratio===null||trial.wait_ratio===undefined?null:trial.wait_ratio*100;
  const infra=trial.infra_failure_rate===null||trial.infra_failure_rate===undefined?null:trial.infra_failure_rate*100;
  const metrics=[
    ['objective',trial.objective_value,trial.objective_name||'not selected'],
    ['rollout tok/s',trial.rollout_tok_s],['train tok/s',trial.train_tok_s],
    ['step time',trial.step_time_s,'s'],['peak VRAM',trial.peak_vram_gb,'GB'],
    ['wait',wait,'%'],['infra failure',infra,'%'],
  ];
  return metrics.map(([label,value,suffix=''])=>`<span class="metric">${esc(label)} <b>${formatNumber(value)}${value===null||value===undefined?'':` ${esc(suffix)}`}</b></span>`).join('');
}
function experimentBrief(detail){
  const manifest=detail.manifest||{};const result=detail.result||{};
  return `<section class="tuning-card"><h3>Experiment brief</h3><dl class="brief"><dt>Hypothesis</dt><dd>${esc(manifest.hypothesis||'Not recorded')}</dd><dt>Observation</dt><dd>${esc(result.observation||'Pending')}</dd><dt>Failure class</dt><dd>${esc(result.failure_class||'None')}</dd><dt>Proposed next step</dt><dd>${esc(result.next_step||'Not recorded')}</dd><dt>Parents</dt><dd>${esc((manifest.parent_trial_ids||[]).join(', ')||'None')}</dd></dl></section>`;
}
function breakdown(detail){
  const durations=detail.metrics?.durations_s||{};const entries=Object.entries(durations).filter(([,value])=>Number.isFinite(Number(value))&&Number(value)>=0);
  if(!entries.length)return `<section class="tuning-card"><h3>Time breakdown</h3><div class="empty compact">No duration breakdown recorded.</div></section>`;
  const max=Math.max(...entries.map(([,value])=>Number(value)),1);
  return `<section class="tuning-card"><h3>Time breakdown</h3><div class="breakdown">${entries.map(([name,value])=>`<div><span>${esc(name)}</span><i style="width:${Math.max(2,Number(value)/max*100)}%"></i><b>${formatNumber(value)}s</b></div>`).join('')}</div></section>`;
}
function jsonCard(title,value){return `<section class="tuning-card"><h3>${esc(title)}</h3><pre>${esc(JSON.stringify(value||{},null,2))}</pre></section>`;}
function comparisonHtml(comparison){
  if(!comparison)return '';
  const trials=comparison.trials||[];const keys=comparison.varying_config_keys||[];
  const rows=keys.map(key=>`<tr><th>${esc(key)}</th>${trials.map(trial=>`<td>${esc(JSON.stringify(trial.config[key]))}</td>`).join('')}</tr>`).join('');
  return `<section class="tuning-card comparison"><h3>Selected config comparison</h3><div class="table-scroll"><table><thead><tr><th>Varying setting</th>${trials.map(trial=>`<th>${esc(trial.trial_id)}</th>`).join('')}</tr></thead><tbody>${rows||'<tr><td colspan="9">Selected configs are identical.</td></tr>'}</tbody></table></div></section>`;
}
function renderTuningDetail(){
  const viewer=$('#tuningViewer');const trial=state.tuning.selected;if(!viewer||!trial)return;
  const detail=state.tuning.detail.get(trial.trial_id);
  if(!detail){viewer.innerHTML='<div class="loading">Loading performance trial…</div>';return;}
  viewer.innerHTML=`${comparisonHtml(state.tuning.comparison)}<div class="tuning-grid">${experimentBrief(detail)}${breakdown(detail)}${jsonCard('Runtime config',detail.config)}${jsonCard('Measured metrics',detail.metrics)}${jsonCard('Profiles',detail.profiles)}${jsonCard('Artifact inventory',detail.artifacts)}${jsonCard('Fingerprints',{hardware:trial.hardware_fingerprint,model:trial.model_fingerprint,workload:trial.workload_fingerprint,software:trial.software_fingerprint,config:trial.config_digest})}</div>`;
}
function bind(){
  document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>{
    setMode(button.dataset.mode);
  }));
  if(state.mode==='performance'){
    document.querySelectorAll('[data-tuning-filter]').forEach(select=>select.addEventListener('change',event=>{
      state.tuning.filters[event.target.dataset.tuningFilter]=event.target.value;state.tuning.page=0;loadTuningTrials();
    }));
    document.querySelectorAll('[data-trial]').forEach(element=>element.addEventListener('click',event=>{
      if(event.target.matches('[data-compare]'))return;
      state.tuning.selected=state.tuning.trials.find(item=>item.trial_id===element.dataset.trial);render();loadTuningDetail();
    }));
    document.querySelectorAll('[data-compare]').forEach(input=>input.addEventListener('change',event=>{
      const selected=state.tuning.compare;if(event.target.checked)selected.add(event.target.dataset.compare);else selected.delete(event.target.dataset.compare);
      state.tuning.comparison=null;render();if(state.tuning.selected)loadTuningDetail();
    }));
    $('#compareTrials')?.addEventListener('click',loadComparison);
    $('#tuningPrev')?.addEventListener('click',()=>{if(state.tuning.page>0){state.tuning.page--;loadTuningTrials();}});
    $('#tuningNext')?.addEventListener('click',()=>{if((state.tuning.page+1)*state.tuning.limit<state.tuning.total){state.tuning.page++;loadTuningTrials();}});
    $('#resetTuningFilters')?.addEventListener('click',()=>{state.tuning.filters={phase:'',status:'',hardware_fingerprint:'',model_fingerprint:'',workload_fingerprint:'',software_fingerprint:'',q:''};state.tuning.page=0;loadTuningTrials();});
    $('#globalSearch')?.addEventListener('input',debounce(event=>{state.tuning.filters.q=event.target.value.trim();state.tuning.page=0;loadTuningTrials();},250));
    $('#themeToggle')?.addEventListener('click',()=>setTheme(theme()==='light'?'dark':'light'));
    return;
  }
  $('[data-filter="run_name"]')?.addEventListener('change',filterChange);
  $('[data-filter="domain"]')?.addEventListener('change',filterChange);
  $('[data-filter="threat_model"]')?.addEventListener('change',filterChange);
  $('[data-filter="status"]')?.addEventListener('change',filterChange);
  document.querySelectorAll('[data-attack]').forEach(b=>b.addEventListener('click',()=>{
    const value=b.dataset.attack;
    state.filters.attack_success=value==='null'?'':value;
    state.filters.attack_evaluated=value==='null'?'false':'';
    state.page=0; loadEpisodes();
  }));
  document.querySelectorAll('[data-episode]').forEach(el=>el.addEventListener('click',()=>{state.selected=state.episodes.find(x=>x.episode_id===el.dataset.episode);state.attempt=null;render();loadDetail();}));
  document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>{state.tab=b.dataset.tab; document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b)); renderDetail();}));
  $('#prev')?.addEventListener('click',()=>{if(state.page>0){state.page--;loadEpisodes();}});
  $('#next')?.addEventListener('click',()=>{if((state.page+1)*state.limit<state.total){state.page++;loadEpisodes();}});
  $('#resetFilters')?.addEventListener('click',()=>{state.filters={run_name:'',domain:'',threat_model:'',status:'',attack_success:'',attack_evaluated:'',q:''};state.page=0;$('#globalSearch').value='';loadEpisodes();});
  $('#globalSearch')?.addEventListener('input',debounce(e=>{state.filters.q=e.target.value.trim();state.page=0;loadEpisodes();},250));
  $('#themeToggle')?.addEventListener('click',()=>setTheme(theme()==='light'?'dark':'light'));
}
function filterChange(e){state.filters[e.target.dataset.filter]=e.target.value;state.page=0;loadEpisodes();}
function debounce(fn,ms){let t;return(...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),ms)}}
function render(){
  const f=state.facets||{};
  const light=theme()==='light';
  const performance=state.mode==='performance';const tf=state.tuning.facets||{};
  const searchValue=performance?state.tuning.filters.q:state.filters.q;
  const stats=performance?`<span><b>${tf.total??0}</b> trials</span><span><b>${tf.successful??0}</b> measured</span>`:`<span><b>${f.total??0}</b> episodes</span><span><b>${f.domains?.length??0}</b> domains</span><span><b>${f.attack_successes??0}</b> attacks</span>`;
  const workspace=performance?`${tuningSidebar()}${tuningListPane()}${tuningDetailShell()}`:`${sidebar()}${listPane()}${detailShell()}`;
  $('#app').innerHTML=`<div class="shell"><header class="topbar"><div class="brand"><div class="logo"></div><div><strong>DTAP Explorer</strong><small>experiment observability</small></div></div><nav class="mode-switch" role="tablist" aria-label="Explorer workspace"><button type="button" role="tab" aria-selected="${!performance}" data-mode="trajectories" class="${performance?'':'active'}">Trajectories</button><button type="button" role="tab" aria-selected="${performance}" data-mode="performance" class="${performance?'active':''}">Performance</button></nav><div class="search"><input id="globalSearch" placeholder="${performance?'Search trial, hardware, model, hypothesis…':'Search task, episode, model, risk category…'}" value="${esc(searchValue)}"></div><div class="statline">${stats}</div><button class="theme-toggle" id="themeToggle" type="button" aria-label="Switch to ${light?'dark':'light'} theme" aria-pressed="${light}">${light?'☾ Dark':'☀ Light'}</button></header><div class="workspace ${performance?'performance-workspace':''}">${workspace}</div></div>`;
  bind(); if(performance)renderTuningDetail();else renderDetail();
}
function showFailure(err){console.error(err);$('#app').innerHTML=`<div class="empty"><div><strong>Explorer failed to load</strong>${esc(err.message)}</div></div>`;}
window.addEventListener('popstate',()=>{
  const next=modeFromLocation();
  if(next!==state.mode)setMode(next,{replace:true});
});
(async()=>{
  await loadFacets();
  await loadEpisodes();
  if(state.mode==='performance')await Promise.all([loadTuningFacets(),loadTuningTrials()]);
})().catch(showFailure);
