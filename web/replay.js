import {InstrumentScene} from '/replay/scene.js';
const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=v=>typeof v==='number'&&Number.isFinite(v);
const fixed=(v,d=2)=>num(v)?v.toFixed(d):'—';
const pretty=v=>typeof v==='string'?v:JSON.stringify(v??null,null,2);
const parse=v=>{try{return typeof v==='string'?JSON.parse(v):v;}catch{return v;}};
const state={data:null,index:0,mode:'all',tab:'request',timer:null,generation:0,detailGeneration:0,event:null,raw:'',details:new Map()};
const names={device_discovery:'Explore instruments',device_selection:'Select an instrument',diagnostic:'Assess the next action',operation_decision:'Choose an action',router:'Route the workflow',procedure:'Prepare the command',evaluator:'Evaluate the result',report:'Final report'};
const actionNames={'workflow.inspect':'Inspect the instrument','workflow.readback':'Read instrument state','experiment.start':'Start experiment recording','experiment.stop':'Stop experiment recording','thermostat.set_tj':'Set jacket temperature','thermostat.set_tr':'Set reactor temperature','thermostat.set_mode':'Set temperature control mode','stirrer.set':'Set agitator speed','power.set':'Set instrument power'};
const guides={device_discovery:'Explore the instrument candidates and their public operations.',device_selection:'Choose an instrument from the proposed candidates.',diagnostic:'Propose the next action using the objective, available readings, and action history.',operation_decision:'Select one of the proposed actions.',router:'Choose the next stage from the selected action, revision, and evidence.',procedure:'Turn the chosen action into an executable instrument command.',evaluator:'Check the recorded outcome against the task conditions.',report:'Record why the workflow stopped.'};
let scene;
try{scene=new InstrumentScene($('scene'),$('scene-status'));scene.load('/replay/model/rc1mx.glb').catch(()=>{});}catch(error){$('scene-status').textContent='WebGL is unavailable in this browser. Messages and recorded states can still be replayed.';}
function modelName(step){return /deepseek/i.test(step.model)?'DeepSeek':/jev/i.test(step.model)?'Jev':step.model||'Model';}
function title(step){if(step.capability==='workflow.await')return `Advance simulation · ${fixed(step.advance_s,0)} s`;if(step.category==='action')return actionNames[step.capability]||actionNames[step.capability?.split('::').pop()]||step.capability||'Instrument action';return names[step.component]||step.kind||'Recorded event';}
function activeSteps(){return(state.data?.steps||[]).filter(s=>state.mode==='all'||s.keyframe);}
function stop(){if(state.timer)clearInterval(state.timer);state.timer=null;$('play').textContent='▶ Play';}
function toast(message){$('toast').textContent=message;$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,2500);}
async function getJSON(path){const r=await fetch(path,{cache:'no-store'});const data=await r.json();if(!r.ok)throw Error(data.error||`HTTP ${r.status}`);return data;}
function reading(step,key){return step.readings[key];}
function value(step,key){return step.controls[key]?.value??step.readings[key]?.value;}
function celsius(slot){return num(slot?.value)?slot.value-273.15:null;}
function heatingState(step){
  const m=step.monitor,power=value(step,'power.enabled')??value(step,'power.on'),emergency=value(step,'emergency.active'),mode=value(step,'thermostat.mode');
  const unavailable={on:null,level:0,label:'Heating unknown',basis:'No heater power or usable temperature target was recorded.',mode};
  if(power===false||emergency===true)return {...unavailable,on:false,label:'Heating disabled',basis:power===false?'Recorded power is off.':'Recorded emergency is active.'};
  if(num(m?.heater_power_w))return {on:m.heater_power_w>0,level:m.heater_power_w>0?Math.min(1,Math.max(.35,m.heater_power_w/2000)):0,label:m.heater_power_w>0?'Recorded heating':'Recorded heater off',basis:`Monitor heater power: ${fixed(m.heater_power_w,1)} W at ${fixed(m.time_s,0)} s.`,mode};
  if(mode==='off')return {...unavailable,on:false,label:'Heating disabled',basis:'Recorded thermostat mode is off.'};
  if(mode!=='tr'&&mode!=='tj')return unavailable;
  const goal=value(step,`thermostat.${mode}_target_k`),setpoint=value(step,`thermostat.${mode}_setpoint_k`),target=num(goal)?goal:num(setpoint)?setpoint:null;
  const slot=reading(step,mode==='tr'?'reactor.indicated_k':'jacket.indicated_k');
  const current=mode==='tr'?(num(m?.true_reactor_c)?m.true_reactor_c:num(m?.reactor_c)?m.reactor_c:celsius(slot)):(num(m?.jacket_c)?m.jacket_c:celsius(slot));
  if(!num(target)||!num(current))return unavailable;
  const targetC=target-273.15,gap=targetC-current,on=gap>.1+1e-9,source=mode==='tr'&&num(m?.true_reactor_c)?'simulator reactor temperature':num(mode==='tr'?m?.reactor_c:m?.jacket_c)?'monitor temperature':`last reading at ${fixed(slot?.time_s,0)} s`;
  return {on,level:on?Math.min(1,Math.max(.4,gap/5)):0,label:on?'Estimated heating':'Estimated idle',target_c:targetC,mode,basis:`${mode.toUpperCase()} ${num(goal)?'accepted target':'last setpoint'} ${fixed(targetC)} °C; ${source} ${fixed(current)} °C. Glow when the gap exceeds 0.10 °C; this is a visual estimate, not recorded heater power.`};
}
function visualState(step){const m=step.monitor,heat=heatingState(step),cal=m?.cal_heater_on??reading(step,'cal_heater.on')?.value,power=value(step,'power.enabled')??value(step,'power.on');return{time_s:step.time_s,reactor_c:num(m?.reactor_c)?m.reactor_c:celsius(reading(step,'reactor.indicated_k')),jacket_c:num(m?.jacket_c)?m.jacket_c:celsius(reading(step,'jacket.indicated_k')),rpm:num(m?.stirrer_rpm)?m.stirrer_rpm:reading(step,'stirrer.indicated_rpm')?.value,power:value(step,'power.enabled')??value(step,'power.on'),emergency:value(step,'emergency.active'),drain_angle:value(step,'drain.angle_rad'),heating_level:heat.level,cal_heating_level:cal===true&&power!==false&&value(step,'emergency.active')!==true?1:0};}
function renderReadings(step){
  const ids=[['Reactor · Tr','reactor.indicated_k','°C'],['Jacket · Tj','jacket.indicated_k','°C']];
  const known=ids.map(x=>reading(step,x[1])).filter(Boolean),stale=known.some(r=>(num(r.time_s)&&r.time_s<step.time_s)||r.revision<step.revision);
  $('reading-badge').textContent=!known.length?'Not read yet':stale?'Earlier reading':'Current reading';$('reading-badge').className='badge'+(stale?' stale':'');
  $('last-readings').innerHTML=ids.map(([label,key,unit])=>{const r=reading(step,key),age=r&&num(r.time_s)?Math.max(0,step.time_s-r.time_s):null,old=r&&(age>0||r.revision<step.revision);return `<div class="last-reading"><div><span>${label}</span><strong>${fixed(celsius(r))} <small>${unit}</small></strong></div><p class="${old?'stale-text':''}">${r?`${esc(r.evidence_id||'Reading')} · at ${fixed(r.time_s,0)} s${age>0?` · ${fixed(age,0)} s old`:old?' · earlier revision':''}`:'No sensor reading has been logged yet.'}</p></div>`;}).join('');
}
function renderTrend(step){
  const samples=state.data.samples.filter(s=>s.time_s<=step.time_s&&num(s.true_reactor_c));
  if(!samples.length){$('trend').innerHTML='<text x="15" y="47">No ground-truth samples available</text>';return;}
  const vals=samples.map(s=>s.true_reactor_c),lo=Math.min(...vals)-.5,hi=Math.max(...vals)+.5,maxT=Math.max(1,step.time_s);
  const x=t=>10+240*t/maxT,y=t=>65-50*(t-lo)/(hi-lo);
  const path=samples.map((s,i)=>`${i?'L':'M'}${x(s.time_s).toFixed(2)},${y(s.true_reactor_c).toFixed(2)}`).join(' ');
  const last=samples.at(-1);
  $('trend').innerHTML=`<path d="M10 65H250" stroke="#dbe6dd" fill="none"/><path d="${path}" stroke="#5a9e7a" stroke-width="1.8" fill="none"/><circle cx="${x(last.time_s)}" cy="${y(last.true_reactor_c)}" r="3" fill="#30865e"/><text x="10" y="81">0 s</text><text x="248" y="81" text-anchor="end">${fixed(step.time_s,0)} s</text><text x="10" y="12">${fixed(Math.max(...vals),1)} °C</text>`;
}
function renderState(step){
  const m=step.monitor,visual=visualState(step),heat=heatingState(step);
  $('heat-status').textContent=heat.label;$('heat-status').className='heat-status'+(heat.on===true?' active':'');$('heat-status').title=heat.basis;
  $('revision').textContent=`rev ${step.revision}`;
  $('true-temp').textContent=fixed(m?.true_reactor_c);
  $('truth-source').textContent=m?`Monitor sample at ${fixed(m.time_s,0)} s · not visible to the agent${m.time_s<step.time_s?' · earlier sample':''}`:'Ground truth was not saved in this run.';
  renderReadings(step);renderTrend(step);
  const mode=value(step,'thermostat.mode'),rpm=visual.rpm,set=value(step,'stirrer.setpoint_rpm'),recording=value(step,'experiment.running');
  const data=[['Agitator',num(rpm)?`${fixed(rpm,0)} rpm`:'Not recorded',num(set)?`Commanded: ${fixed(set,0)} rpm`:''],['Control mode',mode?String(mode).toUpperCase():'Not recorded',step.controls['thermostat.mode']?.source||''],['Recording',recording===true?'Running':recording===false?'Stopped':'Not recorded',step.controls['experiment.running']?.source||''],['Model conversion',num(m?.conversion)?`${fixed(m.conversion*100,2)}%`:'Not recorded','Independent evaluator only']];
  data.push(['Heating',heat.label,heat.mode?`${String(heat.mode).toUpperCase()} temperature control`:''],['Heating target',num(heat.target_c)?`${fixed(heat.target_c)} °C`:'Not recorded','0.10 °C display threshold']);
  $('state-values').innerHTML=data.map(([label,val,note])=>`<div class="state-row"><span>${label}</span><strong>${esc(val)}${note?`<small>${esc(note)}</small>`:''}</strong></div>`).join('');
  $('heat-explanation').textContent=heat.basis;
  $('stage-time').innerHTML=`${fixed(step.time_s,0)} <small>s</small>`;
  $('stage-temp-label').textContent=num(m?.true_reactor_c)?'SIMULATOR TEMPERATURE':'LAST REACTOR READING';
  $('stage-temp').innerHTML=`${fixed(num(m?.true_reactor_c)?m.true_reactor_c:visual.reactor_c)} <small>°C</small>`;
  $('stage-rpm').innerHTML=`${fixed(rpm,0)} <small>rpm</small>`;
  $('scene-phase').textContent=m?.temperature_phase==='holding'?'Holding temperature':m?.temperature_phase==='warmup'?'Warming up':num(step.time_s)?`Recorded state · ${fixed(step.time_s,0)} s`:'No state available';
  return visual;
}
function renderNavigation(){const list=activeSteps(),pos=list.findIndex(s=>s.index===state.index);$('scrubber').max=Math.max(0,list.length-1);$('scrubber').value=Math.max(0,pos);$('position').textContent=`${Math.max(0,pos)+1} / ${list.length} ${state.mode==='all'?'steps':'keyframes'}`;$('prev').disabled=pos<=0;$('next').disabled=pos>=list.length-1;$('play').disabled=list.length<2;$('step-select').value=String(state.index);}
function fillSteps(){const list=activeSteps();$('step-select').innerHTML=list.map(s=>`<option value="${s.index}">${String(s.index+1).padStart(3,'0')} · ${esc(title(s))}</option>`).join('');}
async function select(index,{instant=false}={}){
  if(!state.data?.steps[index])return;
  const previous=state.data.steps[state.index],step=state.data.steps[index];state.index=index;state.event=null;
  $('step-title').textContent=title(step);
  const actor=step.category==='model'?modelName(step):step.category==='action'?'Instrument':step.category==='error'?'Stopped':'Evaluator';
  $('actor-badge').textContent=actor;$('actor-badge').className='badge '+actor.toLowerCase();
  const date=new Date(step.timestamp),wall=Number.isNaN(+date)?'Timestamp unavailable':new Intl.DateTimeFormat('en-AU',{timeZone:'Australia/Sydney',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(date)+' Sydney';
  $('step-meta').textContent=`Step ${index+1} · ${wall} · Simulation ${fixed(step.time_s,0)} s`;
  $('direction').textContent=step.category==='model'?`Workflow → ${actor}`:step.category==='action'?'Workflow → Instrument':'Workflow → Run record';
  const args=Array.isArray(step.arguments.state_ids)?`Read ${step.arguments.state_ids.length} requested state fields.`:pretty(step.arguments);
  $('step-summary').innerHTML=step.category==='action'?`<code>${esc(step.capability)}</code><br>${esc(args)}`:esc(guides[step.component]||'Inspect the saved request and response for this step.');
  $('source-label').textContent=`${step.source.file}:${step.source.line}`;
  const delta=step.time_s-(previous?.time_s??step.time_s),duration=instant?0:Number($('transition').value);
  $('advance-note').textContent=delta!==0?`${delta>0?'+':''}${fixed(delta,0)} s simulated${duration?` → ${(duration/1000).toFixed(1)} s transition`:''}`:'Simulation clock unchanged';
  $('stage-note').textContent=step.advance_s>0?`${fixed(step.advance_s,0)} s wait compressed · agitator continues at recorded RPM.`:'Recorded state · agitator keeps rotating while RPM is above zero.';
  const visual=renderState(step);scene?.transition(visual,duration);renderNavigation();
  const ticket=++state.detailGeneration,run=state.data.summary.id;
  $('message-content').innerHTML='<p>Loading the original message…</p>';
  try{const key=`${run}/${index}`;let detail=state.details.get(key);if(!detail){detail=await getJSON(`/api/replay/run/${encodeURIComponent(run)}/step/${index}`);state.details.set(key,detail);}if(ticket!==state.detailGeneration)return;state.event=detail.event;renderMessage();}catch(error){if(ticket===state.detailGeneration)$('message-content').innerHTML=`<p>${esc(error.message)}</p>`;}
}
function renderMessage(){
  const e=state.event;if(!e)return;const meta=e.llm_metadata||e.backend_metadata||{},request=meta.request||{};
  document.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.tab===state.tab)));
  const actor=modelName(state.data.steps[state.index]);
  let chunks=[],intro='',heading='';
  if(state.tab==='request'){
    heading=request.questions?'Decision request to Jev':meta.request?`Request to ${actor}`:e.proposal?'Command sent to instrument':'Recorded input';
    if(Array.isArray(request.messages))chunks=request.messages.map(m=>({role:m.role==='system'?`Instructions to ${actor}`:m.role==='user'?'Task and instrument context':`${fieldTitle(m.role)} message`,value:parse(m.content),open:m.role!=='system'}));
    else if(request.questions){intro=renderQuestions(request.questions);chunks=[{role:'Public context sent to Jev',value:parse(request.state),open:false},{role:'Original question payload',value:request.questions,open:false}];}
    else chunks=[{role:'Instrument command and parameters',value:e.proposal||e.input||{note:'This event has no model request.'},open:true}];
  }else if(state.tab==='response'){
    heading=request.questions?'Jev decision & candidate comparison':meta.request?`Response from ${actor}`:e.result?'Instrument execution receipt':'Recorded result';
    if(request.questions)intro=renderQuestions(request.questions,meta.response?.answers||{},true);
    if(Array.isArray(meta.response?.choices))chunks=meta.response.choices.map(c=>({role:'Model response',value:parse(c.message?.content),open:true}));
    else chunks=[{role:request.questions?'Original Jev response':e.result?'Execution result and observations':'Recorded response',value:meta.response||e.result||e.output||e.decision||e.report||e.score||e.error,open:!request.questions}];
  }else {heading='Original saved event';chunks=[{role:'Complete event JSON',value:e,open:true}];}
  state.raw=chunks.map(c=>`[${c.role}]\n${pretty(c.value)}`).join('\n\n');
  $('message-content').innerHTML=`<div class="stream-heading"><h3>${esc(heading)}</h3><span>${state.tab==='request'?'SENT':state.tab==='response'?'RECEIVED':'SOURCE'}</span></div>${intro}`+chunks.map(c=>`<details class="message-block" ${c.open?'open':''}><summary>${esc(c.role)}</summary>${state.tab==='event'?jsonBlock(c.value):renderFields(c.value)}</details>`).join('');
  $('message-content').scrollTop=0;
}

const fieldTitles={target:'Task objective',objective:'Objective',observations:'Recorded instrument observations',capabilities:'Available instrument operations',candidate_capabilities:'Candidate operation definitions',previous_hypotheses:'Previous hypotheses',hypotheses:'Proposed explanations',candidates:'Proposed candidate actions',last_error:'Last error',last_verification:'Previous verification',retrieval_evidence:'Retrieved evidence',operation_context:'Recent operation context',operation_memory:'Operation history and receipts',snapshot_status:'Reading freshness',intent:'Action intent',selected_intent:'Selected action',selected_intent_revision:'Selection revision',selected_intent_current:'Selection is current',arguments:'Action parameters',capability_id:'Instrument operation',expected_revision:'Expected state revision',sample_id:'Instrument / sample',request_id:'Request ID',postcondition:'Expected outcome',state:'State sent to model',instructions:'Decision instructions',context:'Decision context',native_response:'Native instrument receipt',measurement:'Sensor readings',advanced_seconds:'Simulation time advanced (seconds)',result:'Execution result',status:'Execution status',error:'Error details',rationale:'Candidate rationale',evidence_ids:'Evidence references',discriminating_observation:'Observation that distinguishes this action',probabilities:'Model-reported probabilities',answers:'Decision answers',usage:'Recorded token usage and cost',candidate_id:'Candidate ID'};
function fieldTitle(key){return fieldTitles[key]||String(key??'Content').replace(/[_-]/g,' ').replace(/^./,c=>c.toUpperCase());}
function highlightedJSON(value){
  const text=typeof value==='string'?value.split(/\r?\n/).map(line=>{const item=parse(line);return item&&typeof item==='object'?pretty(item):line;}).join('\n'):pretty(value);
  const tokens=/"(?:\\.|[^"\\])*"(\s*:)?|\b(?:true|false|null)\b|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/g;
  let output='',cursor=0;
  for(const match of text.matchAll(tokens)){output+=esc(text.slice(cursor,match.index));const token=match[0],type=match[1]?'key':token.startsWith('"')?'string':token==='null'?'null':token==='true'||token==='false'?'boolean':'number';output+=`<span class="json-${type}">${esc(token)}</span>`;cursor=match.index+token.length;}
  return output+esc(text.slice(cursor));
}
function jsonBlock(value){return `<pre class="json-content">${highlightedJSON(value)}</pre>`;}
function renderFields(value){
  if(!value||typeof value!=='object'||Array.isArray(value))return jsonBlock(value);
  return Object.entries(value).map(([key,item])=>`<section class="message-field"><h4>${esc(fieldTitle(key))}</h4>${jsonBlock(item)}</section>`).join('')||'<p class="empty-content">No fields were recorded.</p>';
}
function questionEntries(questions){return Array.isArray(questions)?questions.map((q,i)=>[q.id||String(i),q]):questions&&typeof questions==='object'?Object.entries(questions):[];}
function renderQuestions(questions,answers={},showResult=false){
  return questionEntries(questions).map(([id,question])=>{
    if(!question||typeof question!=='object')return jsonBlock(question);
    const entries=question.criteria&&typeof question.criteria==='object'?Object.entries(question.criteria):[],answer=answers[id]||{};
    const answered=typeof answer.choice==='string'||typeof answer.choice==='number',choice=answered?String(answer.choice):null;
    const title=id==='next_action'?'Choose the next action':fieldTitle(id);
    const result=showResult?`<div class="decision-result"><span>Jev selected</span><strong>${answered?esc(choice):'No answer recorded'}</strong></div>`:'';
    return `<section class="decision-question" aria-label="${esc(title)}"><div class="question-title"><h4>${esc(title)}</h4><span>${entries.length} options</span></div>${question.instructions?`<details class="question-instructions"><summary>Decision instructions</summary><p>${esc(question.instructions)}</p></details>`:''}${result}<h4 class="options-heading">${showResult?'Candidates and reported probabilities':'Candidates sent to Jev'}</h4><div class="candidate-list">${entries.map(([label,raw])=>{
      const data=parse(raw),object=data&&typeof data==='object'&&!Array.isArray(data)?data:null,intent=object?.intent||{},text=typeof data==='string'?data:pretty(data);
      const route=!object?text.match(/^([^:\n]{1,80}):\s*([\s\S]*)$/):null;
      const name=object?.candidate_id||route?.[1]||`Option ${label}`;
      const selected=showResult&&choice===String(label),probability=answer.probabilities?.[label],hasProbability=showResult&&num(probability);
      return `<article class="candidate-card${selected?' selected':''}" aria-label="Option ${esc(label)}: ${esc(name)}"><div class="candidate-title"><span class="option-letter">${esc(label)}</span><strong>${esc(fieldTitle(name))}</strong>${selected?'<span class="selected-mark">Selected</span>':''}</div>${intent.capability_id?`<code class="candidate-operation">${esc(intent.capability_id)}</code>`:''}${hasProbability?`<div class="probability-row"><span>Reported probability</span><strong>${fixed(probability*100,1)}%</strong></div>`:''}<details class="candidate-details"><summary>${object?'Parameters and full rationale':'Full option description'}</summary>${object?renderFields(object):jsonBlock(raw)}</details></article>`;
    }).join('')||'<p class="empty-content">No candidate criteria were saved in this request.</p>'}</div></section>`;
  }).join('');
}
function move(delta){const list=activeSteps(),position=list.findIndex(s=>s.index===state.index),next=list[position+delta];if(next)select(next.index);else stop();}
async function load(run){
  stop();const ticket=++state.generation;++state.detailGeneration;state.data=null;state.event=null;state.details.clear();$('workspace').setAttribute('aria-busy','true');$('page-error').hidden=true;
  $('step-title').textContent='Loading recorded steps';$('message-content').innerHTML='';
  for(const id of ['prev','next','play','step-select','scrubber'])$(id).disabled=true;
  try{const data=await getJSON(`/api/replay/run/${encodeURIComponent(run)}`);if(ticket!==state.generation)return;state.data=data;state.index=0;scene?.setSupported(data.supported_model);$('instrument-title').innerHTML=data.supported_model?'RC1mx <span>+ HFCal</span>':esc(data.summary.device||'Instrument replay');$('original-link').href='/?run='+encodeURIComponent(run);history.replaceState(null,'','/replay?run='+encodeURIComponent(run));$('outcome').textContent=data.summary.success===true?'✓ Complete':data.summary.success===false?'Incomplete':'Not evaluated';fillSteps();$('step-select').disabled=!data.steps.length;$('scrubber').disabled=!data.steps.length;if(data.steps.length)await select(0,{instant:true});else{$('step-title').textContent='No interaction trace';$('message-content').innerHTML='<p>This run has no recorded steps to replay.</p>';}
    if(data.warnings.length)toast(`${data.warnings.length} log warning(s); some records could not be loaded.`);
  }catch(error){if(ticket===state.generation){$('page-error').textContent=error.message;$('page-error').hidden=false;}}finally{if(ticket===state.generation)$('workspace').setAttribute('aria-busy','false');}
}
async function refresh(){try{const data=await getJSON('/api/replay/runs');$('run-select').innerHTML=data.runs.map(r=>`<option value="${esc(r.id)}">${esc(r.id)}${r.success===true?' · Complete':''}</option>`).join('');if(!data.runs.length){$('page-error').textContent='No saved runs found in the project results folder.';$('page-error').hidden=false;return;}const wanted=new URLSearchParams(location.search).get('run')||state.data?.summary.id;const chosen=data.runs.find(r=>r.id===wanted)||data.runs.find(r=>r.device==='mt_rc1mxhfcal')||data.runs[0];$('run-select').value=chosen.id;await load(chosen.id);}catch(error){$('page-error').textContent=error.message;$('page-error').hidden=false;}}
$('refresh').onclick=refresh;$('run-select').onchange=()=>load($('run-select').value);
$('step-select').onchange=()=>{stop();select(Number($('step-select').value));};
$('scrubber').oninput=()=>{stop();const step=activeSteps()[Number($('scrubber').value)];if(step)select(step.index);};
$('prev').onclick=()=>{stop();move(-1);};$('next').onclick=()=>{stop();move(1);};
$('step-mode').onchange=()=>{stop();state.mode=$('step-mode').value;fillSteps();const list=activeSteps();const step=list.find(s=>s.index>=state.index)||list.at(-1);if(step)select(step.index);};
$('play').onclick=()=>{if(state.timer){stop();return;}const list=activeSteps();if(!list.length)return;if(state.index===list.at(-1).index)select(list[0].index);$('play').textContent='Ⅱ Pause';state.timer=setInterval(()=>move(1),2600);};
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{state.tab=b.dataset.tab;renderMessage();});
$('copy-message').onclick=async()=>{try{await navigator.clipboard.writeText(state.raw);toast('Message copied');}catch{toast('Select the message text to copy it.');}};
$('focus-reactor').onclick=()=>scene?.focusReactor();$('reset-camera').onclick=()=>scene?.resetCamera();
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();stop();move(e.key==='ArrowRight'?1:-1);}});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
refresh();
