#!/usr/bin/env python3
"""Turn a benchmark run manifest into a self-contained HTML report.

    python3 scripts/report.py [eval/results/system/latest.json]   ->  report.html beside it

Called automatically at the end of run_benchmark.py. The report EMBEDS the manifest as JSON and
renders the views in-page (ladder / resource / accuracy / provenance / log), so it doubles as a
PLATFORM: add a new view by adding one render function that reads `M` — no re-run needed. Fully
self-contained (no external assets), dark/light theme-aware, same design language as the decks.
"""
import json, os, sys

REPO = "/home/kyle/Documents/GitHub/personal-ai-framework"
DEFAULT = os.path.join(REPO, "eval/results/system/latest.json")

TEMPLATE = r"""<title>Benchmark Report</title>
<style>
  :root{--bg:#f4f6f8;--panel:#fff;--ink:#15202b;--muted:#5b6b7a;--line:#dde4ea;--accent:#e8871e;--accent-ink:#9a5509;
    --meas:#1f9d6b;--meas-bg:#e3f5ec;--deriv:#c98a12;--deriv-bg:#fbf0d8;--src:#7a8794;--src-bg:#eef2f5;
    --hero:#0f1a24;--hero-ink:#eef3f7;--hero-muted:#9fb2c2;--hero-line:#24384a;
    --mono:ui-monospace,"SF Mono",Menlo,"Cascadia Mono",monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
  @media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0c141c;--panel:#121d27;--ink:#eaf1f6;--muted:#9bb0c0;--line:#243441;
    --accent:#f5a632;--accent-ink:#f5a632;--meas:#39c288;--meas-bg:#102a20;--deriv:#e0ad3e;--deriv-bg:#2c2410;--src:#8ea0ad;--src-bg:#1a2630;
    --hero:#080f16;--hero-ink:#eef3f7;--hero-muted:#9fb2c2;--hero-line:#1d2f3e;}}
  :root[data-theme="dark"]{--bg:#0c141c;--panel:#121d27;--ink:#eaf1f6;--muted:#9bb0c0;--line:#243441;--accent:#f5a632;--accent-ink:#f5a632;
    --meas:#39c288;--meas-bg:#102a20;--deriv:#e0ad3e;--deriv-bg:#2c2410;--src:#8ea0ad;--src-bg:#1a2630;--hero:#080f16;--hero-ink:#eef3f7;--hero-muted:#9fb2c2;--hero-line:#1d2f3e;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1100px;margin:0 auto;padding:clamp(20px,4vw,48px)}
  header.hero{background:var(--hero);color:var(--hero-ink);border-radius:14px;padding:28px 30px;margin-bottom:22px;border:1px solid var(--hero-line)}
  .kick{font-family:var(--mono);font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);font-weight:700;margin:0 0 10px}
  h1{font-size:clamp(24px,3.4vw,38px);margin:0 0 8px;letter-spacing:-.02em;font-weight:800}
  .meta{font-family:var(--mono);font-size:13px;color:var(--hero-muted)}
  h2{font-size:19px;letter-spacing:-.01em;margin:26px 0 10px;border-left:3px solid var(--accent);padding-left:10px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:14px 16px}
  .card .big{font-family:var(--mono);font-size:26px;font-weight:800;color:var(--accent-ink);letter-spacing:-.02em}
  .card .lbl{font-size:12px;color:var(--muted);margin-top:4px}
  .tscroll{overflow-x:auto;border:1px solid var(--line);border-radius:11px}
  table{border-collapse:collapse;width:100%;font-size:14px;background:var(--panel)}
  th,td{text-align:left;padding:9px 13px;border-bottom:1px solid var(--line)}
  th{font-family:var(--mono);font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
  td.num,th.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
  tbody tr:last-child td{border-bottom:none} tbody tr:hover{background:color-mix(in srgb,var(--accent) 7%,transparent)}
  .hl{color:var(--accent-ink);font-weight:800}
  .chip{display:inline-block;font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.06em;padding:2px 6px;border-radius:5px;text-transform:uppercase}
  .m{background:var(--meas-bg);color:var(--meas);border:1px solid var(--meas)} .d{background:var(--deriv-bg);color:var(--deriv);border:1px solid var(--deriv)} .s{background:var(--src-bg);color:var(--src);border:1px solid var(--src)}
  .skip{color:var(--muted);font-style:italic;padding:10px 2px;font-size:14px}
  .legend{font-family:var(--mono);font-size:11px;color:var(--muted);margin:14px 0}
  details{margin-top:14px;border:1px solid var(--line);border-radius:10px;background:var(--panel)} summary{cursor:pointer;padding:10px 14px;font-family:var(--mono);font-size:12px;color:var(--muted)}
  pre{margin:0;padding:0 14px 14px;font-family:var(--mono);font-size:12px;color:var(--ink);white-space:pre-wrap}
  .mut{color:var(--muted);font-size:13px}
  tr.clk{cursor:pointer} tr.clk:hover{background:color-mix(in srgb,var(--accent) 9%,transparent)}
  .caret{color:var(--muted);font-family:var(--mono);font-size:11px}
  .det td{background:color-mix(in srgb,var(--accent) 4%,var(--panel));padding:14px 16px}
  .bar{display:flex;height:24px;border-radius:6px;overflow:hidden;margin:4px 0 8px;font-family:var(--mono);font-size:10px;font-weight:700}
  .seg{display:flex;align-items:center;justify-content:center;color:#0b0f14;white-space:nowrap;overflow:hidden}
  .s-comp{background:#8a7fd6} .s-compose{background:var(--accent)} .s-rest{background:#39c288}
  .ovh{font-family:var(--mono);font-weight:800;color:var(--accent-ink)}
</style>
<div class="wrap"><div id="app"></div></div>
<script>
const M = __MANIFEST__;
const CAT = __CATALOG__;
function tLabel(id){const c=CAT[id]; return c?("Test "+c.n+": "+c.name):id;}
function tLink(id){return CAT[id]?('<a href="#test-'+id+'" style="color:inherit;text-decoration:none;border-bottom:1px dotted var(--muted)">'+tLabel(id)+'</a>'):id;}
function tog(id){const e=document.getElementById(id); if(e) e.style.display=(e.style.display==='none'?'table-row':'none');}
function phases(v){
  const wall=(v.wall_s||0)*1000, comp=v.prefill_ms||0, compose=v.decode_ms||0;
  const rest=Math.max(0, wall-comp-compose), llm=comp+compose;
  let tool=''; const ex=v.extra||{};
  for(const k in ex){ if(k.endsWith('_s') && typeof ex[k]==='number') tool+=`${k.replace(/_s$/,'')} ${(ex[k]*1000).toFixed(0)}ms · `; }
  return {wall, comp, compose, rest, pct: wall?Math.round(100*llm/wall):0, tool: tool.replace(/ · $/,'')};
}
function detailRow(id,v){
  const p=phases(v), w=p.wall||1, g=v.gpu||{};
  const seg=(c,ms,l)=> ms>0?`<div class="seg ${c}" style="width:${(100*ms/w).toFixed(1)}%">${100*ms/w>11?l:''}</div>`:'';
  return `<tr class="det" id="det-${id}" style="display:none"><td colspan="7">`+
    `<div class="bar">${seg('s-comp',p.comp,'comprehend')}${seg('s-compose',p.compose,'compose')}${seg('s-rest',p.rest,'work + orchestrate')}</div>`+
    `<div class="mut">comprehend (analyze prompt+context, prefill) <b>${p.comp.toFixed(0)}ms</b> · compose (write answer, decode) <b>${p.compose.toFixed(0)}ms</b> · work + orchestrate (tool run, loop glue) <b>${p.rest.toFixed(0)}ms</b> &nbsp;—&nbsp; <span class="ovh">LLM overhead ${p.pct}%</span>${p.tool?' · measured tool time: '+p.tool:''}</div>`+
    `<div class="mut" style="margin-top:6px">GPU footprint: sm ${num(g.sm_pct_mean)}% · mem ${num(g.mem_pct_mean)}% · ${num(g.power_w_mean)} W · peak VRAM ${num(g.vram_mib_peak)} MiB</div>`+
    (v.output_head?`<div class="mut" style="margin-top:6px;font-family:var(--mono);font-size:11px">out: ${E(v.output_head).replace(/</g,'&lt;')}</div>`:'')+
    `</td></tr>`;
}
const E=(s)=> (s==null?"":String(s));
const num=(v)=> (v==null?"&mdash;":(typeof v==="number"?(Math.round(v*100)/100):v));
function provChip(p){p=E(p).toUpperCase(); if(p.startsWith("MEAS"))return '<span class="chip m">measured</span>';
  if(p.startsWith("DERIV"))return '<span class="chip d">derived</span>'; if(p.startsWith("SOURC")||p.startsWith("SRC"))return '<span class="chip s">sourced</span>'; return '<span class="chip s">'+E(p).slice(0,10)+'</span>';}

// ---- views (add a function here to add a view; it reads M) ----
function vHeader(){const r=M.run||{};return `<header class="hero"><p class="kick">agentic-edge benchmark · report</p>
  <h1>Run ${E(r.ts)}</h1><div class="meta">model: ${E((M.resource||{}).model||"&mdash;")} &nbsp;·&nbsp; boards: ${(r.boards||[]).join(", ")} &nbsp;·&nbsp; scope: ${E(r.scope||"")}</div></header>`;}

function vHeadline(){const res=M.resource||{},lad=M.ladder||{},acc=M.accuracy||{};let c=[];
  const sr=(res.tasks||{}).spec_rag; if(sr)c.push(['5090 decode',num(sr.decode_tok_s)+' t/s','spec_rag, harness']);
  if(Object.keys(lad).length)c.push(['boards on ladder',Object.keys(lad).length,'decode measured']);
  if(res.tasks)c.push(['tasks profiled',Object.keys(res.tasks).length,'10-task harness']);
  const ag=(acc.agreement||{}); if(ag.agree!=null)c.push(['two-judge agree',ag.agree+' / '+(ag.agree+(ag.split||0)),'Sonnet + GPT-4o']);
  if(!c.length)return ''; return `<div class="cards">`+c.map(x=>`<div class="card"><div class="big">${x[1]}</div><div class="lbl">${x[0]}</div><div class="mut" style="font-size:11px">${x[2]}</div></div>`).join('')+`</div>`;}

function vLadder(){const lad=M.ladder; if(!lad||!Object.keys(lad).length)return '<h2>Decode ladder</h2><div class="skip">not run this pass</div>';
  let rows=Object.entries(lad).map(([b,v])=>`<tr><td>${b}</td><td>${E(v.model||'7B')}</td><td class="num hl">${num(v.decode_tok_s)}</td><td class="num">${num(v.prefill_tok_s)}</td><td>${E(v.backend)}</td><td>${provChip(v.prov)}</td></tr>`).join('');
  return `<h2>Decode ladder</h2><div class="tscroll"><table><thead><tr><th>board</th><th>model</th><th class="num">decode t/s</th><th class="num">prefill t/s</th><th>backend</th><th>prov</th></tr></thead><tbody>${rows}</tbody></table></div>`;}

function vResource(){const res=M.resource; if(!res||!res.tasks)return '<h2>Workload — per-task resource</h2><div class="skip">not run this pass (Skippy down?)</div>';
  let rows=Object.entries(res.tasks).map(([t,v])=>{const g=v.gpu||{};return `<tr class="clk" onclick="tog('det-${t}')"><td>${tLink(t)} <span class="caret">▸ phases</span></td><td class="num">${num(v.wall_s)}</td><td class="num">${num(v.decode_tok_s)}</td><td class="num">${num(v.prefill_tok_s)}</td><td class="num">${num(g.sm_pct_mean)}</td><td class="num">${num(g.mem_pct_mean)}</td><td class="num">${num(g.power_w_mean)}</td></tr>`+detailRow(t,v);}).join('');
  return `<h2>Workload — per-task resource <span class="chip m">measured</span></h2><div class="mut" style="margin-bottom:6px">Click any row for its phase breakdown — <b>comprehend → work → compose</b> — and the <b>agent overhead</b> (the LLM's share of the time vs. the actual work). <span style="font-size:11px">Caveat: phase split comes from /generate telemetry; tasks that run the LLM via the agent-loop or an upload endpoint (multi-tool chain, meeting-summary) don't emit it, so their LLM time falls into work+orchestrate (shows as 0% — a telemetry gap, not truly 0).</span></div><div class="tscroll"><table><thead><tr><th>task</th><th class="num">wall s</th><th class="num">decode t/s</th><th class="num">prefill t/s</th><th class="num">sm%</th><th class="num">mem%</th><th class="num">power W</th></tr></thead><tbody>${rows}</tbody></table></div>`;}

function vAccuracy(){const acc=M.accuracy; if(!acc||!acc.results)return '<h2>Task-success — two-judge</h2><div class="skip">not graded this pass (no API keys?)</div>';
  let rows=Object.entries(acc.results).map(([t,v])=>{const s=(v.sonnet||{}).verdict,g=(v.gpt4o||{}).verdict;return `<tr><td>${tLink(t)}</td><td>${E(s)}</td><td>${E(g)}</td><td>${v.agree?'&check;':'<b style="color:var(--accent-ink)">split</b>'}</td></tr>`;}).join('');
  const ag=acc.agreement||{};
  return `<h2>Task-success — two-judge (Sonnet + GPT-4o) <span class="chip m">measured</span></h2><div class="mut">agree ${ag.agree} / split ${ag.split} &nbsp;·&nbsp; splits flagged, never averaged</div><div class="tscroll" style="margin-top:8px"><table><thead><tr><th>task</th><th>sonnet</th><th>gpt-4o</th><th>agree?</th></tr></thead><tbody>${rows}</tbody></table></div>`;}

function vProvenance(){const p=M.provenance||{};return `<h2>Provenance</h2><div class="legend"><span class="chip m">measured</span> ran it on a censused box &nbsp; <span class="chip d">derived</span> computed from measurements &nbsp; <span class="chip s">sourced</span> another team / vendor</div>`+
  `<div class="tscroll"><table><tbody>`+Object.entries(p).map(([k,v])=>`<tr><td>${k}</td><td class="mut">${E(v)}</td></tr>`).join('')+`</tbody></table></div>`;}

function vCatalog(){const ids=Object.keys(CAT).sort((a,b)=>CAT[a].n-CAT[b].n); if(!ids.length)return '';
  return `<h2>Test descriptions</h2><div class="mut" style="margin-bottom:8px">What each test actually does (the task names above link here).</div><div class="tscroll"><table><thead><tr><th class="num">#</th><th>test</th><th>what it does</th><th>stresses</th></tr></thead><tbody>`+
    ids.map(id=>`<tr id="test-${id}"><td class="num">${CAT[id].n}</td><td><b>${E(CAT[id].name)}</b></td><td class="mut">${E(CAT[id].desc)}</td><td class="mut">${E(CAT[id].stresses)}</td></tr>`).join('')+`</tbody></table></div>`;}

function vLog(){if(!M.log)return '';return `<details><summary>run log (${M.log.length} lines)</summary><pre>${M.log.map(E).join("\n").replace(/</g,"&lt;")}</pre></details>`;}

document.getElementById('app').innerHTML = [vHeader(),vHeadline(),vLadder(),vResource(),vAccuracy(),vProvenance(),vCatalog(),vLog()].join('');
</script>
"""

def build_report(manifest_path=DEFAULT, out_html=None):
    man = json.load(open(manifest_path))
    cat_path = os.path.join(REPO, "eval/test_catalog.json")
    cat = json.load(open(cat_path)) if os.path.exists(cat_path) else {}
    cat = {k: v for k, v in cat.items() if not k.startswith("_")}
    html = TEMPLATE.replace("__MANIFEST__", json.dumps(man)).replace("__CATALOG__", json.dumps(cat))
    out = out_html or os.path.join(os.path.dirname(manifest_path), "report.html")
    open(out, "w").write(html)
    return out

if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print("wrote", build_report(mp))
