'use strict';
const $ = id => document.getElementById(id);
const state = {runs: [], data: null, selected: 0, filter: 'all', search: '', view: 'conversation', timer: null, rawTab: 'request', generation: 0};
const phases = [['discover','Choose instrument'],['diagnose','Assess state'],['route','Choose next step'],['prepare','Build command'],['execute','Execute and wait'],['verify','Verify result']];
const componentNames = {device_discovery:'Device discovery',device_selection:'Device selection',diagnostic:'Assessment',router:'Workflow routing',operation_decision:'Action selection',procedure:'Command generation',execute:'Instrument execution',verification:'Result verification',report:'Workflow report',evaluator:'Independent evaluation',gate:'Action validation'};
const routeNames = {diagnose:'Assess current state',prepare_action:'Build execution command',verify:'Verify task result',wait_short:'Short wait',wait_long:'Long wait',stop:'Stop task',deliberate:'Reassess'};
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const pretty = value => typeof value === 'string' ? value : JSON.stringify(value ?? null, null, 2);
const fmt = value => typeof value === 'number' ? value.toLocaleString('en-AU') : '—';
const boolText = value => value === true ? 'Yes' : value === false ? 'No' : 'Not recorded';
const meta = e => e.llm_metadata || e.backend_metadata || {};
const runName = id => { const m = id.match(/(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/); return m ? `${m[2]}/${m[3]} · ${m[4]}:${m[5]}:${m[6]}` : id; };
const clock = value => { if (!value) return 'Time not recorded'; const date = new Date(value); return Number.isNaN(+date) ? value : new Intl.DateTimeFormat('en-AU',{timeZone:'Australia/Sydney',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(date); };
const modelName = e => { const m = meta(e); const id = m.model_id || m.request?.model || m.backend_id || ''; return /jev/i.test(id) ? 'Jev' : /deepseek/i.test(id) ? 'DeepSeek' : id || 'Model'; };
const phase = e => ['device_discovery','device_selection'].includes(e.component) ? 'discover' : e.component === 'diagnostic' ? 'diagnose' : ['router','operation_decision'].includes(e.component) ? 'route' : ['procedure','gate'].includes(e.component) ? 'prepare' : e.component === 'execute' ? 'execute' : 'verify';
function actionName(cap, args = {}) {
  if (!cap) return 'Action not recorded';
  if (cap === 'workflow.inspect') return 'Read initial instrument state';
  if (cap === 'workflow.readback') return 'Read current state';
  if (cap === 'workflow.await') return `Advance simulation by ${args.duration_s ?? '?'} s`;
  if (cap.endsWith('::power.set')) return args.on === true ? 'Power on' : args.on === false ? 'Power off' : 'Set power';
  if (cap.endsWith('::timer.set')) return `Set timer to ${Number(args.days || 0)*86400+Number(args.hours || 0)*3600+Number(args.minutes || 0)*60} s`;
  return cap.split('::').pop();
}
function title(entry) {
  const e = entry.event, o = e.output || {};
  if (e.kind === 'role_error') return 'Model output validation failed · Stopped';
  if (e.kind === 'operation_error') return 'Action choice validation failed · Stopped';
  if (e.component === 'device_discovery') return ({search:'Search for an instrument',describe:'Inspect instrument capabilities',propose:'Propose instrument candidates',stop:'Stop device search'})[o.action] || 'Explore the device directory';
  if (e.component === 'device_selection') return 'Select an instrument';
  if (e.component === 'router') return routeNames[e.decision?.choice] || ({wait_30:'Wait 30 s',wait_120:'Wait 120 s'})[e.decision?.choice] || `Choose: ${e.decision?.choice || 'Next step'}`;
  if (e.component === 'diagnostic') return `Assess state · ${actionName(o.candidates?.[0]?.intent?.capability_id,o.candidates?.[0]?.intent?.arguments)}`;
  if (e.component === 'operation_decision') return 'Select a proposed action';
  if (e.component === 'procedure') return actionName(o.capability_id,o.arguments);
  if (e.kind === 'tool_result') return actionName(e.proposal?.capability_id,e.proposal?.arguments);
  if (e.component === 'verification') return 'Model checks final evidence';
  if (e.kind === 'agent_report') return 'Generate final report';
  if (e.kind === 'instrument_score') return 'Independently check task completion';
  return componentNames[e.component] || e.kind || 'Record';
}
function categoryName(entry) {return entry.category === 'model' ? modelName(entry.event) : entry.category === 'action' ? 'Simulated instrument' : entry.category === 'error' ? 'Error record' : 'Evaluation result';}
function parseContent(value) { if (typeof value !== 'string') return value; try {return JSON.parse(value);} catch {return value;} }
function outputFor(e) {return e.output ?? e.decision ?? e.report ?? e.score ?? e.result ?? parseContent(meta(e).response?.choices?.[0]?.message?.content) ?? meta(e).response ?? {};}
function nativeIn(value, depth = 0) {
  if (!value || typeof value !== 'object' || depth > 5) return null;
  if (value.observations && typeof value.observations === 'object' && (value.clock || value.instance_id)) return value;
  for (const key of ['measurement','native','readback','result','observations']) { const found=nativeIn(value[key],depth+1); if(found) return found; }
  return null;
}
function nativeAt(index) {const selected=state.data.entries[index];if(selected?.snapshot?.native)return {native:selected.snapshot.native,id:selected.snapshot.id,index:selected.snapshot_step??index,source:selected.snapshot_source};for(let i=index;i>=0;i--){const e=state.data.entries[i].event;const native=nativeIn(e.result);if(native)return {native,id:e.result?.measurement?.id || e.result?.id || '',index:i};}return null;}
function inputSummary(e) {
  switch(e.component){
    case 'device_discovery': return 'Search and inspect the public device directory, then propose instruments for the objective and required outcomes.';
    case 'device_selection': return 'Choose an instrument from the candidates proposed by the expert.';
    case 'diagnostic': return 'Use available readings, capability contracts, and context to propose verifiable next actions.';
    case 'operation_decision': return 'Choose the next action from the expert candidates using the available state and task conditions.';
    case 'procedure': return 'Convert the selected action into an executable command, preserving the instrument, arguments, state revision, and request ID.';
    case 'router': return 'Use task progress, the selected action, evidence freshness, and remaining budget to choose the next workflow stage.';
    case 'verification': return 'Check the objective against available instrument readings, cite evidence IDs, and assess completion.';
    default:return 'Inspect the recorded input and result for this step.';
  }
}
function argumentsHTML(args) {return Object.entries(args || {}).length ? `<div class="argument-list">${Object.entries(args).map(([k,v])=>`<span class="argument">${esc(k)} = ${esc(typeof v==='object'?JSON.stringify(v):v)}</span>`).join('')}</div>` : '';}
function operationHTML(intent) { if(!intent)return '';return `<strong>${esc(actionName(intent.capability_id,intent.arguments))}</strong><div><span class="code-tag">${esc(intent.capability_id)}</span></div>${argumentsHTML(intent.arguments)}`; }
function probabilitiesHTML(e) {
  const p=e.decision?.probabilities || {}, chosen=e.decision?.choice;
  if(!Object.keys(p).length)return '';
  return `<div class="answer-block">${Object.entries(p).map(([key,value])=>`<div class="probability"><span>${key===chosen?'✓ ':''}${esc(routeNames[key]||key)}</span><span class="prob-track"><i style="width:${Math.max(0,Math.min(100,Number(value)*100))}%"></i></span><span class="prob-value">${(Number(value)*100).toFixed(1)}%</span></div>`).join('')}<p class="prob-note">These model scores compare candidates. They need not sum to 100% and are not experiment success probabilities.</p></div>`;
}
function responseHTML(e,entry) {
  const o=outputFor(e);
  if(e.kind==='operation_error'){
    const answer=meta(e).response?.answers?.next_action, p=answer?.probabilities;
    return `<p><strong>The action choice failed validation. The candidate was not executed.</strong></p><p>The workflow stopped without an automatic retry.</p><pre class="raw-output">${esc(e.error)}</pre>${p?`<p>Original choice: ${esc(answer.choice)} · Score sum: ${Object.values(p).reduce((sum,v)=>sum+Number(v),0).toFixed(4)}</p><pre class="raw-output">${esc(pretty(p))}</pre>`:''}<p class="muted">The raw response below preserves the API result.</p>`;
  }
  if(e.kind==='role_error')return `<p><strong>The model response failed contract validation. The action was not executed.</strong></p><p>${e.will_repair?'The workflow permits one format repair.':'This run stopped without an automatic retry.'}</p><pre class="raw-output">${esc(e.error)}</pre><p class="muted">The raw response below preserves the original model output.</p>`;
  if(e.component==='device_discovery'){
    if(o.action==='search')return `<p>Search the device directory for:</p><span class="code-tag">${esc(o.query)}</span>`;
    if(o.action==='describe')return `<p>Inspect capabilities and state fields for these instruments:</p>${(o.device_ids||[]).map(id=>`<span class="code-tag">${esc(id)}</span>`).join(' ')}`;
    if(o.action==='propose')return (o.candidates||[]).map(c=>`<p>Proposed instrument: <span class="code-tag">${esc(c.device_id)}</span></p><div class="guide-label">Model rationale · Original text</div><p class="english">${esc(c.rationale)}</p>`).join('');
  }
  if(e.component==='diagnostic')return (o.candidates||[]).map(c=>`<div>${operationHTML(c.intent)}<div class="answer-block"><div class="guide-label">Model rationale · Original text</div><p class="english">${esc(c.rationale)}</p>${(c.evidence_ids||[]).map(id=>`<span class="mini-status">Evidence ${esc(id)}</span>`).join('')}</div></div>`).join('<div class="answer-block"></div>') || `<p>${esc(pretty(o))}</p>`;
  if(e.component==='procedure')return `${operationHTML(o)}<div class="answer-block"><div class="guide-label">Expected outcome · Original model text</div><p class="english">${esc(o.postcondition)}</p></div>`;
  if(['router','operation_decision','device_selection'].includes(e.component)){
    let desc=routeNames[o.choice]||o.choice;
    if(e.component==='operation_decision' && e.selected)desc=actionName(e.selected.intent?.capability_id,e.selected.intent?.arguments);
    return `<p>Selected <strong>${esc(desc)}</strong></p>${e.selected?.intent?operationHTML(e.selected.intent):''}${probabilitiesHTML(e)}`;
  }
  if(e.component==='verification')return `<p>${o.status==='supported'?'The available evidence supports task completion.':o.status==='contradicted'?'The available evidence contradicts task completion.':'The available evidence is insufficient to confirm completion.'}</p><span class="code-tag">status: ${esc(o.status)}</span><div class="argument-list">${(o.evidence_ids||[]).map(id=>`<span class="argument">Cited evidence: ${esc(id)}</span>`).join('')}</div>${o.missing_evidence?.length?`<p>${esc(pretty(o.missing_evidence))}</p>`:''}`;
  if(e.kind==='tool_result'){
    const n=entry?.snapshot?.native||nativeIn(e.result),timer=n?.observations?.['timer.state']?.value;
    return `<p>The instrument returned a record for this action${n?.status?` with status <span class="code-tag">${esc(n.status)}</span>`:''}.</p>${timer?`<p>Timer setting: <strong>${esc(timer.set_s??'Not recorded')} s</strong>; timer expired: <strong>${boolText(timer.expired)}</strong>.</p>`:''}<p class="muted">The full instrument receipt is available in the raw record below.</p>`;
  }
  if(e.kind==='agent_report' && o.terminal_reason==='task_completed')return `<p><strong>The host monitor detected all completion conditions and stopped the task.</strong></p><p>Success is determined by the independent evaluator. The immediate-stop policy makes no additional model verification call.</p><p>Original model report status: <span class="code-tag">${esc(o.status)}</span> · Stop reason: <span class="code-tag">task_completed</span></p>`;
  if(e.kind==='agent_report')return `<p>Report status: <strong>${esc(({supported:'Goal supported by evidence',contradicted:'Goal not met',unresolved:'Unresolved'})[o.status]||o.status)}</strong></p><p>Stop reason: <span class="code-tag">${esc(o.terminal_reason)}</span></p><p>Cited evidence: ${esc((o.evidence_ids||[]).join(', ')||'Not recorded')}</p>`;
  if(e.kind==='instrument_score')return `<p><strong>${o.task_success===true?'The task passed independent evaluation.':'The task did not pass all evaluation checks.'}</strong></p><p>Goal met: ${boolText(o.goal_met)}<br>Protocol valid: ${boolText(o.protocol_correct)}<br>Fresh evidence: ${boolText(o.fresh_evidence)}<br>Pending operation: ${o.pending_operation?esc(pretty(o.pending_operation)):'None'}</p>`;
  return `<pre class="raw-output">${esc(pretty(o))}</pre>`;
}
function filteredEntries() {return (state.data?.entries||[]).filter(entry=>{const matches=state.filter==='all'||state.filter==='action'&&entry.category==='action'||state.filter==='deepseek'&&entry.category==='model'&&modelName(entry.event)==='DeepSeek'||state.filter==='jev'&&entry.category==='model'&&modelName(entry.event)==='Jev';return matches&&(!state.search||entry._search.includes(state.search));});}
function renderTimeline(scroll=false) {
  const rows=filteredEntries();
  $('timeline').innerHTML=rows.length?rows.map(entry=>`<button class="step ${entry.index===state.selected?'selected':''}" data-index="${entry.index}" data-category="${esc(entry.category)}" aria-current="${entry.index===state.selected?'step':'false'}"><span class="step-number">${String(entry.index+1).padStart(2,'0')}</span><span class="step-main"><span class="step-title">${esc(title(entry))}</span><span class="step-meta"><span>${esc(categoryName(entry))}</span><span>${clock(entry.event.timestamp)}</span></span></span><i class="step-indicator"></i></button>`).join(''):'<div class="empty-search">No matching interactions. Try a different filter.</div>';
  if(scroll)$('timeline').querySelector('.selected')?.scrollIntoView({block:'nearest',inline:'nearest',behavior:'smooth'});
}
function snapshotHTML() {
  const found=nativeAt(state.selected);
  if(!found)return '<div class="native-snapshot"><div class="snapshot-title">Instrument state associated with this step</div><p class="snapshot-note">No instrument readings yet. Device selections and model responses are not instrument observations.</p></div>';
  const obs=found.native.observations||{},timer=obs['timer.state']?.value||{};
  const values=[['Simulation time',found.native.clock?.time_s,'s'],['Power state',obs['power.on']?.value===true?'On':obs['power.on']?.value===false?'Off':undefined,''],['Timer setting',timer.set_s,'s'],['Timer expired',timer.expired===undefined?undefined:boolText(timer.expired),''],['Data quality',obs['timer.state']?.quality,''],['Evidence ID',found.id||undefined,'']];
  return `<div class="native-snapshot"><div class="snapshot-title"><span>Latest instrument reading associated with this step</span><span>From step ${found.index+1}</span></div><div class="snapshot-grid">${values.map(([label,value,unit])=>`<div>${label}<strong>${esc(value??'Not recorded')}${value!==undefined&&unit?' '+unit:''}</strong></div>`).join('')}</div>${found.source?`<p class="snapshot-note">Reading source: ${esc(found.source.file)} : ${esc(found.source.line)} · ${clock(found.source.timestamp)}</p>`:''}</div>`;
}
function rawValue(entry,tab) {
  const e=entry.event,m=meta(e);
  if(tab==='readback')return pretty(entry.snapshot||{note:'No reading is associated with this step.'});
  if(tab==='request'){
    if(m.request?.messages)return m.request.messages.map(message=>`[${message.role}]\n${pretty(parseContent(message.content))}`).join('\n\n');
    if(m.request?.state)return `${m.request.state}\n\n[questions]\n${pretty(m.request.questions)}`;
    return pretty(e.proposal||e.input||{note:'This event has no model request.'});
  }
  if(tab==='response'){
    if(m.response?.choices)return m.response.choices.map(choice=>`[${choice.message?.role||'assistant'}]\n${pretty(choice.message?.content)}`).join('\n\n');
    return pretty(m.response||e.result||e.score||e.report||e.output||e.decision);
  }
  return pretty(e);
}
function renderInteraction() {
  const entry=state.data.entries[state.selected];
  if(!entry){$('interaction').innerHTML='<div class="empty">This record contains a result but no interaction trace. Model conversations cannot be reconstructed from result.json alone.</div>';return;}
  const e=entry.event,m=meta(e),isModel=entry.category==='model',name=categoryName(entry);
  const cost=typeof m.usage?.cost==='number'?` · $${m.usage.cost.toFixed(5)}`:'';
  const duration=typeof m.elapsed_ms==='number'?` · ${(m.elapsed_ms/1000).toFixed(2)}s`:'';
  const cls=name==='DeepSeek'?'expert':name==='Jev'?'router':'instrument';
  const author=isModel?`${name} → Workflow`:entry.category==='action'?'Simulated instrument → Workflow':'Final record';
  const question=entry.category==='action'?operationHTML(e.proposal):isModel?`<div class="guide-label">Request guide · Purpose of this call</div><p>${esc(inputSummary(e))}</p>`:'<p>Summarize execution results and evaluate the original task conditions.</p>';
  $('interaction').innerHTML=`<div class="interaction-heading"><div><h2>${esc(title(entry))}</h2><p>Step ${state.selected+1} / ${state.data.entries.length} · ${clock(e.timestamp)} Sydney time${duration}${cost}</p></div><span class="stage-label">${esc(componentNames[e.component]||e.component)}</span></div><div class="message question"><span class="avatar">W</span><div class="message-body"><div class="message-author"><span>${isModel?`Workflow → ${esc(name)}`:entry.category==='action'?'Workflow → Instrument':'Task evaluation'}</span><span>${isModel?'Request':'Input'}</span></div><div class="bubble">${question}</div></div></div><div class="message answer"><span class="avatar ${cls}">${name==='DeepSeek'?'D':name==='Jev'?'J':entry.category==='action'?'I':'✓'}</span><div class="message-body"><div class="message-author"><span>${esc(author)}</span><span>${isModel?'Recorded model response':entry.category==='action'?'Recorded execution receipt':'Result'}</span></div><div class="bubble">${responseHTML(e,entry)}</div></div></div>${snapshotHTML()}<details class="raw-panel"><summary>Expand raw requests and responses</summary><div class="raw-tabs"><button data-raw="request" class="${state.rawTab==='request'?'active':''}">Raw request / question</button><button data-raw="response" class="${state.rawTab==='response'?'active':''}">Raw response</button><button data-raw="readback" class="${state.rawTab==='readback'?'active':''}">Associated reading</button><button data-raw="event" class="${state.rawTab==='event'?'active':''}">Event JSON</button></div><pre id="raw-output" class="raw-output"></pre><div class="raw-footer"><span>${esc(e._source)} : ${esc(e._line)} · Sensitive fields are redacted</span><button class="text-button" id="copy-raw">Copy raw text</button></div></details>`;
  $('raw-output').textContent=rawValue(entry,state.rawTab);
  $('copy-raw').onclick=()=>copyText($('raw-output').textContent);
  $('interaction').querySelectorAll('[data-raw]').forEach(button=>button.onclick=()=>{state.rawTab=button.dataset.raw;$('raw-output').textContent=rawValue(entry,state.rawTab);$('interaction').querySelectorAll('[data-raw]').forEach(b=>b.classList.toggle('active',b===button));});
}
function selectStep(index,scroll=false) {
  if(!state.data?.entries.length)return;
  state.selected=Math.max(0,Math.min(index,state.data.entries.length-1));
  $('scrubber').value=state.selected;$('position').textContent=`${state.selected+1} / ${state.data.entries.length}`;
  const list=filteredEntries(),pos=list.findIndex(e=>e.index===state.selected);
  $('prev').disabled=pos<=0;$('next').disabled=pos===list.length-1||pos<0;
  renderTimeline(scroll);renderInteraction();
  $('flow').querySelectorAll('[data-phase]').forEach(button=>button.classList.toggle('active',button.dataset.phase===phase(state.data.entries[state.selected].event)));
}
function stopPlayback(){if(state.timer)clearInterval(state.timer);state.timer=null;$('play').textContent='▶ Play log';}
function move(delta){const list=filteredEntries(),pos=list.findIndex(e=>e.index===state.selected);const next=list[pos+delta];if(next)selectStep(next.index,true);else stopPlayback();}
function setView(view){state.view=view;stopPlayback();for(const id of ['conversation','actions','json'])$(id+'-view').hidden=id!==view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-selected',b.dataset.view===view));}
function renderActions(){const actions=state.data.entries.filter(e=>e.category==='action');$('actions-view').innerHTML=`<h2 style="font-size:19px;margin:0 0 5px">Executed actions</h2><p class="method-note">Proposed actions pass validation before execution. These ${actions.length} logged actions include controls, reads, and simulation waits.</p><div class="action-list">${actions.map((entry,i)=>`<div class="action-row"><span class="action-num">${i+1}</span><div><h3>${esc(title(entry))}</h3><span class="code-tag">${esc(entry.event.proposal?.capability_id)}</span>${argumentsHTML(entry.event.proposal?.arguments)}<p style="margin-top:7px">${clock(entry.event.timestamp)} · Recorded step ${esc(entry.event.step)}</p></div><button class="text-button" data-jump="${entry.index}">View conversation and receipt →</button></div>`).join('')||'<p class="method-note">No instrument execution events are available in this record.</p>'}</div>`;}
function renderTree(value,parent,key='result',depth=0){
  if(value!==null&&typeof value==='object'){
    const entries=Object.entries(value),details=document.createElement('details');details.className='json-node';details.open=depth<1;const summary=document.createElement('summary');summary.textContent=`${key} ${Array.isArray(value)?'[':'{'} ${entries.length} ${Array.isArray(value)?'items':'fields'} ${Array.isArray(value)?']':'}'}`;details.append(summary);for(const [k,v]of entries)renderTree(v,details,k,depth+1);parent.append(details);
  }else{const line=document.createElement('div');line.className='json-leaf';const name=document.createElement('span');name.className='json-key';name.textContent=key+': ';const val=document.createElement('span');val.className='json-value';val.textContent=JSON.stringify(value);line.append(name,val);parent.append(line);}
}
function renderJSON(){const r=state.data.result,s=r.score||{},p=r.report||{};const guides=[['score.task_success',boolText(s.task_success),'Whether the independent evaluator confirmed task completion from instrument state.'],['report.status',p.status||'Not recorded','The model conclusion. A supported claim must still be checked against the independent score.'],['score.fresh_evidence',boolText(s.fresh_evidence),'Whether final verification used fresh readings after the action completed.'],['score.protocol_correct',boolText(s.protocol_correct),'Whether calls, arguments, and receipts complied with the execution protocol.'],['report.action_count',fmt(p.action_count),'Budgeted actions, including controls and reads; not the number of native invoke calls.'],['report.llm_call_count',fmt(p.llm_call_count),'Total model calls across expert, routing, and action-selection roles.'],['score.simulated_seconds',fmt(s.simulated_seconds)+' s','Simulated instrument time, separate from elapsed wall-clock time.'],['reference_scripts_used',boolText(r.reference_scripts_used),'Whether the run used a prewritten reference solution script. False means none was used.']];$('field-guide').innerHTML=guides.map(([key,value,description])=>`<div class="field-card"><div class="field-top"><code>${key}</code><strong>${esc(value)}</strong></div><p>${description}</p></div>`).join('');$('json-tree').replaceChildren();renderTree(r,$('json-tree'));}
function renderEvidence(){const score=state.data.result.score||{};const conditions=[...(score.predicates||[]),...(score.constraints||[])];$('evidence').innerHTML=conditions.map(p=>{const path=[p.state_id,...(p.path||[])].join('.'),friendly=path==='timer.state.expired'?'Timer has expired':path==='timer.state.set_s'?'Timer is set to 60 s':path;return `<div class="evidence-card"><span class="check ${p.met!==true?'fail':''}">${p.met===true?'✓':'!'}</span><div><strong>${esc(friendly)}</strong><p><span class="code-tag">${esc(path)}</span></p><p>Expected ${esc(pretty(p.expected))} · Actual ${esc(pretty(p.actual))}${p.error?' · '+esc(p.error):''}</p></div></div>`;}).join('')+`<div class="evidence-details"><span>Protocol valid: ${boolText(score.protocol_correct)}</span><span>Fresh evidence: ${boolText(score.fresh_evidence)}</span><span>Lifecycle resolved: ${boolText(score.lifecycle_resolved)}</span></div>`;}
function renderRun(){
  const {summary:s,result:r,task}=state.data;
  const binder=s.task_id==='binder-timer-expiration',reaction=s.device==='mt_rc1mxhfcal';
  $('task-title').textContent=binder?'Oven timer experiment':reaction?'RC1mx + HFCal · Acetic anhydride hydrolysis':s.task_id;
  $('task-description').textContent=task.objective||'Explore model interactions, instrument actions, and final evaluation for this run.';
  $('original-goal').textContent=task.objective||'The original objective was not saved in this record.';
  $('outcome').className='outcome'+(s.success===false?' failure':'');$('outcome').innerHTML=`<strong>${s.success===true?'✓ Task complete':s.success===false?'Task incomplete':'Not evaluated'}</strong><span>Final run outcome</span>`;
  const metrics=[['Wall-clock time',typeof s.elapsed_seconds==='number'?s.elapsed_seconds.toFixed(1):'—','s',`Simulation advanced ${fmt(r.score?.simulated_seconds)} s`],['Model calls',fmt(s.model_calls),'calls',`DeepSeek + Jev · ${fmt(s.total_tokens)} tokens`],['API cost',s.cost_usd==null?'—':'$'+s.cost_usd.toFixed(4),'USD',s.cost_complete?'Costs recorded for all calls':'Includes only returned cost data'],['Instrument',s.device==='binder_fd056'?'Binder FD 56':s.device==='mt_rc1mxhfcal'?'RC1mx + HFCal':s.device||'Not recorded','',r.selection?`Selected from ${r.selection.catalog_count??'—'} instruments`:'Instrument specified by the task']];
  $('metrics').innerHTML=metrics.map(([label,value,unit,note])=>`<div class="metric"><div class="metric-label">${label}</div><div class="metric-value" ${label==='Instrument'?'style="font-size:21px;line-height:1.98"':''}>${esc(value)}<small>${unit}</small></div><div class="metric-note">${esc(note)}</div></div>`).join('');
  $('flow').innerHTML=phases.map(([id,label],i)=>`<button class="flow-node" data-phase="${id}"><span>${i+1}</span>${label}</button>`).join('');
  $('scrubber').max=Math.max(0,state.data.entries.length-1);$('play').disabled=!state.data.entries.length;$('scrubber').disabled=!state.data.entries.length;
  $('source-path').textContent='Record location: '+state.data.files.directory;
  renderActions();renderJSON();renderEvidence();renderRuns();selectStep(0);setView('conversation');
  if(!state.data.entries.length)renderInteraction();
  $('workspace').hidden=false;$('empty').hidden=true;
  if(state.data.warnings.length)toast(state.data.warnings.join(' '));
}
function renderRuns(){$('run-list').innerHTML=state.runs.map(run=>`<button class="run-card ${state.data?.summary.id===run.id?'active':''}" data-run="${esc(run.id)}"><strong>${esc(runName(run.id))}</strong><span class="run-id">${esc(run.task_id)}</span><span class="run-row"><span>${fmt(run.model_calls)} model calls</span><span class="tiny-status ${run.success===false?'failed':''}">${run.success===true?'● Complete':run.success===false?'● Incomplete':'○ Not evaluated'}</span></span></button>`).join('');}
async function getJSON(path){const response=await fetch(path,{cache:'no-store'});const data=await response.json();if(!response.ok)throw Error(data.error||'Failed to load the record');return data;}
async function loadRun(id){const generation=++state.generation;stopPlayback();$('loading').hidden=false;$('workspace').hidden=true;$('error').hidden=true;try{const data=await getJSON('/api/runs/'+encodeURIComponent(id));if(generation!==state.generation)return;state.data=data;state.filter='all';state.search='';state.rawTab='request';$('search').value='';document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter==='all'));data.entries.forEach(entry=>entry._search=(title(entry)+' '+categoryName(entry)+' '+JSON.stringify(entry.event)).toLowerCase());renderRun();}catch(error){if(generation===state.generation){$('error').textContent=error.message;$('error').hidden=false;}}finally{if(generation===state.generation)$('loading').hidden=true;}}
async function refresh(){try{const data=await getJSON('/api/runs');state.runs=data.runs;renderRuns();if(!state.runs.length){$('loading').hidden=true;$('empty').hidden=false;$('workspace').hidden=true;return;}const requested=new URLSearchParams(location.search).get('run');await loadRun(state.runs.some(r=>r.id===requested)?requested:state.runs.some(r=>r.id===state.data?.summary.id)?state.data.summary.id:state.runs[0].id);}catch(error){$('loading').hidden=true;$('error').textContent='Cannot connect to the local service: '+error.message;$('error').hidden=false;}}
function toast(message){$('toast').textContent=message;$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,3500);}
async function copyText(value){try{await navigator.clipboard.writeText(value);toast('Copied');}catch{toast('Clipboard access was denied. Select and copy the raw text manually.');}}
function jump(index){stopPlayback();state.filter='all';state.search='';$('search').value='';document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter==='all'));setView('conversation');selectStep(index,true);}
$('refresh').onclick=refresh;
$('run-list').onclick=event=>{const button=event.target.closest('[data-run]');if(button)loadRun(button.dataset.run);};
$('timeline').onclick=event=>{const button=event.target.closest('[data-index]');if(button){stopPlayback();selectStep(Number(button.dataset.index));}};
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>setView(button.dataset.view));
$('filters').onclick=event=>{const button=event.target.closest('[data-filter]');if(!button)return;stopPlayback();state.filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b===button));const rows=filteredEntries();if(rows.length)selectStep(rows[0].index);else renderTimeline();};
$('search').oninput=()=>{stopPlayback();state.search=$('search').value.toLowerCase().trim();const rows=filteredEntries();if(rows.length)selectStep(rows[0].index);else renderTimeline();};
$('flow').onclick=event=>{const button=event.target.closest('[data-phase]');if(!button)return;const entry=state.data.entries.find(e=>phase(e.event)===button.dataset.phase);if(entry)jump(entry.index);else toast('This stage was not recorded in this run.');};
$('prev').onclick=()=>{stopPlayback();move(-1);};$('next').onclick=()=>{stopPlayback();move(1);};$('last').onclick=()=>jump(state.data.entries.length-1);
$('scrubber').oninput=()=>jump(Number($('scrubber').value));
$('play').onclick=()=>{if(state.timer){stopPlayback();return;}const list=filteredEntries();if(!list.length)return;if(state.selected===list.at(-1).index)selectStep(list[0].index);$('play').textContent='Ⅱ Pause playback';state.timer=setInterval(()=>move(1),1800);};
$('actions-view').onclick=event=>{const button=event.target.closest('[data-jump]');if(button)jump(Number(button.dataset.jump));};
$('copy-result').onclick=()=>copyText(pretty(state.data.result));
$('export').onclick=()=>{const url=URL.createObjectURL(new Blob([pretty(state.data.result)],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=state.data.summary.id+'-result.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
refresh();
