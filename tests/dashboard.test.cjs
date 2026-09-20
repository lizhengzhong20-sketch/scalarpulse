const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../src/scalarpulse/static/index.html'), 'utf8');

function setup() {
  const requests = [], timers = new Map(), notices = [];
  let timerId = 0;
  const element = () => ({textContent:'', innerHTML:'', classList:{toggle(){}}, setAttribute(){}});
  const context = {
    localStorage:{getItem:()=>null}, MAX_POINTS_PER_METRIC:20000, MAX_PAUSED_EVENTS:5000,
    API_STATE:'/api/state', API_EVENTS:'/api/events', scheduleRender(){},
    setConnection(){}, showToast:message=>notices.push(message), console:{warn(){}},
    dom:{emptyTitle:element(),emptyCopy:element(),pauseButton:element(),pauseLabel:element(),pauseIcon:element()},
    EventSource:{OPEN:1},
    setTimeout(fn){timers.set(++timerId,fn); return timerId;},
    clearTimeout(id){timers.delete(id);},
    fetch(url){return new Promise((resolve,reject)=>requests.push({url,resolve,reject}));}
  };
  const code = html.slice(html.indexOf('      const app = {'), html.indexOf('      function scheduleRender()'))
    + html.slice(html.indexOf('      function handleEvent('), html.indexOf('      function csvEscape('));
  const api = vm.runInNewContext(code+'\n({app,normalizedRun,ingestSnapshot,applyPayload,loadSnapshot,handleEvent,togglePause})', context);
  return {...api, requests, timers, notices,
    async flushTimers(){ const callbacks=[...timers.values()]; timers.clear(); callbacks.forEach(fn=>fn()); await tick(); }
  };
}
const tick = () => new Promise(resolve=>setImmediate(resolve));
const meta = {id:'a',name:'A',status:'running',started_at:'2026-01-01T00:00:00Z'};
const record = (seq,value,time=1767225601) => ({run_id:'a',seq,step:1,time,metrics:{loss:value}});
const snapshot = (id='a',records=[]) => ({runs:[meta,{...meta,id:'b',name:'B'}],selected_run_id:id,records});
function respond(request,payload){request.resolve({ok:true,json:async()=>payload});}

test('nested metric names survive history loading', () => {
  const s=setup();
  const run=s.normalizedRun({...meta,records:[{step:1,metrics:{epoch:3,time:4,x:5,loss:6}}]});
  assert.deepEqual([...run.metrics.keys()],['epoch','time','x','loss']);
});
test('late snapshot cannot overwrite a higher sequence at the same step', () => {
  const s=setup(); s.ingestSnapshot(snapshot('a',[record(0,1)]));
  s.applyPayload(record(1,9)); s.ingestSnapshot(snapshot('a',[record(0,1)]));
  assert.equal(s.app.runs.get('a').metrics.get('loss')[0].value,9);
});
test('late live events cannot overwrite a higher sequence either', () => {
  const s=setup(); s.applyPayload(record(2,9)); s.applyPayload(record(1,1));
  assert.equal(s.app.runs.get('a').metrics.get('loss')[0].value,9);
});
test('old snapshot cannot restore running after completion', async () => {
  const s=setup(); s.applyPayload(meta); const pending=s.loadSnapshot('a');
  s.applyPayload({...meta,status:'completed',ended_at:'2026-01-01T00:10:00Z'});
  respond(s.requests[0],snapshot()); await pending;
  assert.equal(s.app.runs.get('a').status,'completed');
});
test('completed duration uses ended_at despite a late metric', () => {
  const s=setup(); s.applyPayload(meta); s.applyPayload(record(0,1));
  s.applyPayload({...meta,status:'completed',ended_at:'2026-01-01T00:10:00Z'});
  s.applyPayload(record(1,2));
  assert.equal(s.app.runs.get('a').updatedAt,'2026-01-01T00:10:00.000Z');
});
test('out of order requests cannot change latest selection', async () => {
  const s=setup(); s.app.selectedRunId='a'; const a=s.loadSnapshot('a');
  s.app.selectedRunId='b'; const b=s.loadSnapshot('b');
  respond(s.requests[1],snapshot('b')); await b;
  respond(s.requests[0],snapshot('a')); await a;
  assert.equal(s.app.selectedRunId,'b');
});
test('old failed request cannot replace a successful current state', async () => {
  const s=setup(); const a=s.loadSnapshot('a'); const b=s.loadSnapshot('b');
  respond(s.requests[1],snapshot('b')); await b;
  s.requests[0].reject(new Error('offline')); await a;
  assert.equal(s.app.fetchFailed,false);
});
test('reset reloads missing history without inventing a default run', async () => {
  const s=setup(); s.app.selectedRunId='a';
  s.handleEvent({type:'reset',data:'{"reason":"overflow"}'});
  await s.flushTimers();
  assert.equal(s.requests.length,1);
  respond(s.requests[0],snapshot('a',[record(0,7)])); await tick();
  assert.equal(s.app.runs.get('a').metrics.get('loss')[0].value,7);
  assert.equal(s.app.runs.has('default'),false);
});
test('paused reset waits for resume, failed resync retries', async () => {
  const s=setup(); s.togglePause();
  s.handleEvent({type:'reset',data:'{"reason":"overflow"}'});
  await s.flushTimers(); assert.equal(s.requests.length,0);
  s.togglePause(); await s.flushTimers(); assert.equal(s.requests.length,1);
  s.requests[0].reject(new Error('offline')); await tick();
  await s.flushTimers(); assert.equal(s.requests.length,2);
  respond(s.requests[1],snapshot('a',[record(1,8)])); await tick();
  assert.equal(s.app.runs.get('a').metrics.get('loss')[0].value,8);
});
test('overflowing pause buffer resyncs rather than claiming complete recovery', async () => {
  const s=setup(); s.togglePause();
  for(let i=0;i<5001;i++) s.handleEvent({type:'metric',data:JSON.stringify(record(i,i))});
  s.togglePause(); await s.flushTimers();
  assert.equal(s.requests.length,1);
  assert.equal(s.notices.some(message=>message.includes('已补齐')),false);
});
test('another reset during recovery triggers another snapshot', async () => {
  const s=setup(); s.handleEvent({type:'reset',data:'{}'}); await s.flushTimers();
  assert.equal(s.requests.length,1);
  s.handleEvent({type:'reset',data:'{}'});
  respond(s.requests[0],snapshot()); await tick(); await s.flushTimers();
  assert.equal(s.requests.length,2);
});

test('metric arriving during fetch does not hide completed snapshot metadata', async () => {
  const s=setup(); s.applyPayload(meta);
  const request=s.loadSnapshot('a');
  s.applyPayload(record(1,9));
  const completed={...meta,status:'completed',ended_at:'2026-01-01T00:10:00Z'};
  respond(s.requests[0],{runs:[completed],run:completed,selected_run_id:'a',records:[record(0,1)]});
  await request;
  assert.equal(s.app.runs.get('a').status,'completed');
  assert.equal(s.app.runs.get('a').metrics.get('loss')[0].value,9);
});

test('legacy same-time snapshot preserves live point received during fetch', async () => {
  const s=setup(); s.applyPayload(meta);
  const request=s.loadSnapshot('a');
  s.applyPayload({...record(0,9),seq:undefined});
  respond(s.requests[0],snapshot('a',[{...record(0,1),seq:undefined}]));
  await request;
  assert.equal(s.app.runs.get('a').metrics.get('loss')[0].value,9);
});

test('selection made while paused invalidates an earlier pending snapshot', async () => {
  const s=setup(); s.ingestSnapshot(snapshot());
  const pending=s.loadSnapshot('a');
  s.togglePause(); s.app.selectedRunId='b'; await s.loadSnapshot('b');
  s.togglePause();
  respond(s.requests[0],snapshot('a')); await pending;
  assert.equal(s.app.selectedRunId,'b');
  await s.flushTimers();
  assert.equal(s.requests[1].url,'/api/state?run_id=b');
});
