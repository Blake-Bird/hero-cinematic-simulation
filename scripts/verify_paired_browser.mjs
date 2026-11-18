// Exercise the atlas as served by scripts/serve_results.py using installed Chrome.
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';

const [origin, report, chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'] = process.argv.slice(2);
if (!origin || !report) throw new Error('Usage: node verify_paired_browser.mjs ORIGIN REPORT [CHROME]');
const profile = await mkdtemp(join(tmpdir(), 'atlas-chrome-'));
const processChrome = spawn(chrome, ['--headless', '--disable-gpu', '--no-first-run',
  '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank'], {stdio: 'ignore'});
let ws;
const result = {origin, films: [], pages: [], failures: []};
try {
  let port;
  for (let i = 0; i < 100 && !port; i++) {
    port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8').catch(() => '')).split('\n')[0];
    if (!port) await delay(100);
  }
  const targets = await (await fetch('http://127.0.0.1:' + port + '/json/list')).json();
  const target = targets.find(t => t.type === 'page' && t.url === 'about:blank') || targets.find(t => t.type === 'page');
  if (!target) throw new Error('Headless Chrome did not expose a page target');
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {ws.addEventListener('open', resolve); ws.addEventListener('error', reject);});
  let id = 0;
  const pending = new Map();
  ws.addEventListener('message', ({data}) => {
    const reply = JSON.parse(data);
    if (!pending.has(reply.id)) return;
    const [resolve, reject] = pending.get(reply.id);
    pending.delete(reply.id);
    if (reply.error) reject(new Error(reply.error.message));
    else resolve(reply.result);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    pending.set(++id, [resolve, reject]);
    ws.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const reply = await send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    if (reply.exceptionDetails) throw new Error(reply.exceptionDetails.exception?.description || reply.exceptionDetails.text);
    return reply.result.value;
  };
  const go = async pathname => {
    const url = new URL(pathname, origin).href;
    let navigation;
    for(let attempt=0;attempt<3;attempt++){
      navigation=await send('Page.navigate',{url});
      if(navigation.errorText!=='net::ERR_ABORTED')break;
      await delay(500);
    }
    if (navigation.errorText) {
      const state=await evaluate('({href:location.href,ready:document.readyState,title:document.title,body:document.body?.innerText.slice(0,300)})');
      if(state.href!==url)throw new Error('Navigation failed: '+navigation.errorText+' '+url+' '+JSON.stringify(state));
    }
    for (let i = 0; i < 100; i++) {
      if (await evaluate('document.readyState === "complete" && location.href === ' + JSON.stringify(url)).catch(() => false)) { await delay(150); return; }
      await delay(100);
    }
    const state=await evaluate('({href:location.href,ready:document.readyState,title:document.title,body:document.body?.innerText.slice(0,400)})');
    throw new Error('Timed out loading ' + url + ' ' + JSON.stringify(state));
  };
  await send('Page.enable');
  await send('Runtime.enable');
  await delay(1000);
  await send('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await go('/04_Movies/index.html');
  const films = await evaluate('[...document.querySelectorAll("[data-film]")].map(x => x.dataset.film)');
  if (!films.length) throw new Error('The library has no playable entries');
  for (const film of films) {
    try {
      const movie = await evaluate(`(async () => {
        const v = document.querySelector('video');
        const button = [...document.querySelectorAll('[data-film]')].find(x => x.dataset.film === ${JSON.stringify(film)});
        function waitFor(event, action) {
          return new Promise((resolve, reject) => {
            const done = () => {clearTimeout(timer); v.removeEventListener('error', fail); resolve();};
            const fail = () => {clearTimeout(timer); v.removeEventListener(event, done); reject(new Error(v.error?.message || event));};
            const timer = setTimeout(fail, 10000);
            v.addEventListener(event, done, {once:true});v.addEventListener('error', fail, {once:true});action();
          });
        }
        await waitFor('loadedmetadata', () => button.click());
        const seekTo = v.duration * .55;
        await waitFor('seeked', () => v.currentTime = seekTo);
        if (Math.abs(v.currentTime - seekTo) > .25 || v.readyState < 2) throw new Error('Seek yielded no decoded frame');
        return {duration: v.duration, seek: v.currentTime, width: v.videoWidth, height: v.videoHeight};
      })()`);
      result.films.push({film, ...movie});
      console.log('PLAYABLE', film);
    } catch (error) {
      result.failures.push({film, error: error.message});
    }
  }
  const stories = await evaluate('[...document.querySelectorAll("a")].filter(a=>a.textContent.includes("interactive merger tree")).map(a=>a.pathname)');
  for (const story of stories) {
    await go(story);
    const state = await evaluate(`(async () => {
      const slider = document.querySelector('input[type=range]');
      slider.value=slider.min;slider.dispatchEvent(new Event('input'));
      await new Promise(r=>setTimeout(r,500));
      const detail=document.querySelector('#detail').textContent;
      if (!detail.includes('exact catalogue ID') || !document.querySelector('.tree circle')) throw new Error('Missing tree identity');
      return detail;
    })()`);
    result.pages.push({story, state});
  }
  for (const model of ['EDE', 'LCDM']) {
    const group = model === 'EDE' ? '02_EDE' : '03_LCDM';
    const epoch = model + '_z_0014p400_snap_021';
    const pathname = '/' + group + '/Saved_epochs/' + epoch + '/' + epoch + '_Interactive_3D.html';
    await go(pathname);
    const state = await evaluate(`(() => {
      const c=document.querySelector('#cloud');
      const before=c.toDataURL();
      c.dispatchEvent(new WheelEvent('wheel', {deltaY: -120, cancelable: true}));
      if (c.width < 200 || c.height < 200 || c.toDataURL() === before) throw new Error('Particle canvas did not redraw on zoom '+JSON.stringify({width:c.width,height:c.height,handler:typeof c.onwheel,changed:c.toDataURL()!==before}));
      return {width:c.width,height:c.height};
    })()`);
    result.pages.push({pathname, ...state});
  }
  for (const path of ['/02_EDE/index.html','/03_LCDM/index.html']) {
    await go(path);
    const state=await evaluate(`(() => {
      const select=document.querySelector('#epoch-scope');
      const visible=()=>[...document.querySelectorAll('a[data-priority]')].filter(x=>!x.hidden).length;
      const initial=visible();
      select.value='all';select.dispatchEvent(new Event('change'));
      const all=visible();
      if(initial!==8 || all!==86) throw new Error('Epoch selector did not expose eight priority / 86 all');
      const search=document.querySelector('#search');
      search.value='snapshot 0';search.dispatchEvent(new Event('input'));
      if(visible()!==1) throw new Error('Saved snapshot search did not locate epoch zero');
      return {initial,all,search:visible()};
    })()`);
    result.pages.push({path,selector:state});
  }
  await go('/04_Movies/index.html');
  const filter = await evaluate(`(() => {
    const search = document.querySelector('#search');
    search.value='does-not-exist';search.dispatchEvent(new Event('input'));
    const empty=document.querySelectorAll('#film-cards article:not([hidden])').length;
    search.value='LCDM';search.dispatchEvent(new Event('input'));
    const results=[...document.querySelectorAll('#film-cards article:not([hidden])')];
    if(empty || !results.length || results.some(x => !x.dataset.search.includes('LCDM'))) throw new Error('Film search did not apply');
    return {results:results.length};
  })()`);
  result.pages.push({pathname:'/04_Movies/index.html', filter});
  for (const path of ['/01_Comparisons/Fitting_function_comparisons/index.html','/02_EDE/Fitting_function_comparisons/index.html','/03_LCDM/Fitting_function_comparisons/index.html']) {
    await go(path);
    const entries=await evaluate('document.querySelectorAll(".grid > .card").length');
    if(entries!==8) throw new Error('Expected eight fitting-function epochs: '+path);
    result.pages.push({path,fittingEpochs:entries});
  }
  for (const path of ['/Spatial_explorers.html','/Halo_histories.html']){
    await go(path);
    const state=await evaluate('({models:document.querySelectorAll(".grid > a.card").length,heading:document.querySelector("h1").textContent})');
    if(state.models!==2)throw new Error('Direct topic hub must expose both models');
    result.pages.push({path,...state});
  }
  for(const model of ['EDE','LCDM'])for(const z of ['014p400','000p000']){
    const section=model==='EDE'?'02_EDE':'03_LCDM';
    const path=`/${section}/Spatial_explorers/${model}_z_${z}/Most_massive_FOF_halo_anatomy/Readable_cutaway/index.html`;
    await go(path);
    const state=await evaluate(`(async()=>{
      const s=document.querySelector('#depth'),im=document.querySelector('#slice');
      const initial=+s.value;s.value=0;s.dispatchEvent(new Event('input'));
      await im.decode();
      if(im.naturalWidth!==768 || !document.querySelector('#depth-label').textContent.includes('occupied display cells'))throw new Error('Cutaway image or occupancy label failed');
      return {initialSlice:initial,width:im.naturalWidth};
    })()`);
    result.pages.push({path,cutaway:state});
  }
  await go('/04_Movies/index.html?q=flow');
  const flow=await evaluate('[...document.querySelectorAll("#film-cards article:not([hidden])")].length');
  if(flow!==6) throw new Error('Expected six flow films, found '+flow);
  result.pages.push({path:'/04_Movies/index.html?q=flow',flowFilms:flow});
  await send('Emulation.setDeviceMetricsOverride', {width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  for (const [pathname, file] of [['/START_HERE.html','atlas_desktop.png'],['/04_Movies/index.html','film_library_desktop.png']]) {
    await go(pathname);
    await delay(250);
    const {data} = await send('Page.captureScreenshot', {format:'png'});
    await writeFile(join(dirname(report),file),Buffer.from(data,'base64'));
  }
  await send('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  for (const path of ['/START_HERE.html', '/02_EDE/index.html', '/03_LCDM/index.html', '/04_Movies/index.html','/01_Comparisons/Fitting_function_comparisons/index.html','/Spatial_explorers.html','/Halo_histories.html']) {
    await go(path);
    const metrics = await evaluate('({heading: document.querySelector("h1").textContent, viewport: innerWidth, content: document.documentElement.scrollWidth})');
    if (metrics.content > metrics.viewport + 2) result.failures.push({path, error: 'Mobile horizontal overflow'});
    result.pages.push({path, ...metrics});
    if (path === '/START_HERE.html') {
      const {data} = await send('Page.captureScreenshot',{format:'png'});
      await writeFile(join(dirname(report),'atlas_mobile.png'),Buffer.from(data,'base64'));
    }
  }
} catch (error) {
  result.failures.push({error: error.message});
} finally {
  ws?.close();
  processChrome.kill();
  await delay(500);
  await rm(profile, {force: true, recursive: true}).catch(() => {});
  await writeFile(report, JSON.stringify(result, null, 2) + '\n');
  console.log(`Checked ${result.films.length} films, ${result.pages.length} pages, ${result.failures.length} failures`);
  if (result.failures.length) process.exitCode = 1;
}
