"use strict";
/*
 * 页内 LLM 核心的数值检查。
 *
 * 核心代码不是复制品：每次运行都用 tools/extract_page_js.js 从 index.html 里
 * 现场抽出内联的 LLM 核心函数，所以页面里的实现就是被测实现。
 *
 * 覆盖：旧行为黄金基线（训练 / 前向 / 生成逐位一致）、nucleus 采样、
 * 滑动窗口掩码、中心有限差分梯度、困惑度、存档往返与非法存档。
 */
const fs = require("fs");
const path = require("path");
const extract = require("./extract_page_js.js");
const { createReporter, summarize } = require("./report.js");

const BASELINE_PATH = path.join(__dirname, "fixtures", "llm_baseline.json");

function eqArr(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}
function eqDeep(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/* 从参数的嵌套结构里收集所有标量叶子（连同它所在的父容器与键） */
function collectScalars(node, gnode, label, out, parent, key) {
  if (Array.isArray(node)) {
    for (let i = 0; i < node.length; i++) {
      collectScalars(node[i], gnode[i], label + "[" + i + "]", out, node, i);
    }
  } else if (node && typeof node === "object") {
    for (const k in node) collectScalars(node[k], gnode[k], label + "." + k, out, node, k);
  } else {
    out.push({ path: label, parent, key, grad: gnode });
  }
}

async function run() {
  const r = createReporter();
  const html = extract.readIndexHtml();
  const core = extract.extractLlmCore(html);
  const L = extract.loadLlmCore(html);
  const baseline = JSON.parse(fs.readFileSync(BASELINE_PATH, "utf8"));

  r.section("0. 从 index.html 抽取核心");
  r.ok("抽出 LLM 核心（不复制代码）", core.code.length > 10000,
    "第 " + core.startLine + "–" + core.endLine + " 行, " + core.code.length + " 字符");
  r.ok("核心导出全部 " + extract.LLM_CORE_FUNCTIONS.length + " 个函数",
    extract.LLM_CORE_FUNCTIONS.every(n => typeof L[n] === "function"));

  /* ---------------- 1. 基线：逐位复现改动前的训练 / 前向 / 生成 ---------------- */
  r.section("1. 旧行为基线（top_p=1.0、window=0 必须逐位一致）");
  const mb = L.llmCreate({ vocabSize: 5, dModel: 4, heads: 2, dff: 6, maxLen: 6, rand: L.llmSeedRng(12345) });
  const idsB = [1, 2, 3, 4, 1];
  const losses = [];
  for (let i = 0; i < 20; i++) losses.push(L.llmTrainStep(mb, idsB, 0.05));
  r.ok("20 步训练 loss 与基线逐位相同", eqArr(losses, baseline.losses), "last=" + losses[19]);
  const fwdB = L.llmForward(mb, [0, 1, 2, 3, 4]);
  r.ok("forward loss 与基线逐位相同", fwdB.loss === baseline.forwardLoss, "loss=" + fwdB.loss);
  r.ok("forward logits 与基线逐位相同", eqDeep(fwdB.logits, baseline.logits));

  const genA = L.llmGenerate(mb, [1, 2], 12, 0.8, 0, L.llmSeedRng(7));
  const genA1 = L.llmGenerate(mb, [1, 2], 12, 0.8, 0, L.llmSeedRng(7), 1.0);
  const genB = L.llmGenerate(mb, [1], 10, 1.0, 3, L.llmSeedRng(99));
  const genC = L.llmGenerate(mb, [0], 16, 1.3, 0, L.llmSeedRng(2024));
  r.ok("top_p 省略（旧 6 参数调用）与基线相同", eqArr(genA, baseline.genA), genA.join(","));
  r.ok("显式 top_p=1.0 与基线相同", eqArr(genA1, baseline.genA), genA1.join(","));
  r.ok("topK=3 / top_p=1.0 与基线相同", eqArr(genB, baseline.genB), genB.join(","));
  r.ok("温度 1.3 / top_p=1.0 与基线相同", eqArr(genC, baseline.genC), genC.join(","));

  const mb0 = L.llmCreate({ vocabSize: 5, dModel: 4, heads: 2, dff: 6, maxLen: 6, window: 0, rand: L.llmSeedRng(12345) });
  for (let i = 0; i < 20; i++) L.llmTrainStep(mb0, idsB, 0.05);
  const genW0 = L.llmGenerate(mb0, [1, 2], 12, 0.8, 0, L.llmSeedRng(7));
  r.ok("显式 window=0 与基线相同（含 20 步训练路径）", eqArr(genW0, baseline.genA), genW0.join(","));
  r.ok("window=0 forward logits 与基线逐位相同",
    eqDeep(L.llmForward(mb0, [0, 1, 2, 3, 4]).logits, baseline.logits));

  /* ---------------- 2. top_p 语义 ---------------- */
  r.section("2. nucleus（top_p）采样");
  const rng = L.llmSeedRng(4242);
  const peaked = [Math.log(0.7), Math.log(0.2), Math.log(0.1)];
  let allTop = true;
  for (let i = 0; i < 200; i++) if (L.llmSampleToken(peaked, 1, 0, 0.5, rng) !== 0) allTop = false;
  r.ok("top_p=0.5 在 0.7/0.2/0.1 分布上只保留最高概率 token", allTop);

  let noTail = true, sawSecond = false;
  for (let i = 0; i < 300; i++) {
    const t = L.llmSampleToken(peaked, 1, 0, 0.8, rng);   /* 0.7+0.2=0.9 ≥ 0.8 */
    if (t === 2) noTail = false;
    if (t === 1) sawSecond = true;
  }
  r.ok("top_p=0.8 保留前两名、丢掉尾部", noTail && sawSecond);

  /* top_p=1.0 与旧采样实现逐位一致（多组随机 logits + 多种温度 / topK） */
  function oldSample(logits, temperature, topK, rand) {
    const V = logits.length;
    let row = logits.map(v => v / temperature);
    if (topK > 0 && topK < V) {
      const sorted = row.slice().sort((x, y) => y - x);
      const cut = sorted[topK - 1];
      row = row.map(v => (v >= cut ? v : -Infinity));
    }
    const probs = L.softmaxRow(row);
    let u = rand(), pick = V - 1;
    for (let i = 0; i < V; i++) { u -= probs[i]; if (u <= 0) { pick = i; break; } }
    return pick;
  }
  let samplerSame = true;
  const lrng = L.llmSeedRng(777);
  for (let trial = 0; trial < 100; trial++) {
    const V = 2 + (trial % 7);
    const logits = Array.from({ length: V }, () => (lrng() - 0.5) * 12);
    const temp = [0.3, 0.8, 1.0, 1.7][trial % 4];
    const topK = [0, 1, 2, 5][trial % 4];
    const a = L.llmSampleToken(logits, temp, topK, 1.0, L.llmSeedRng(trial + 1));
    const b = oldSample(logits, temp, topK, L.llmSeedRng(trial + 1));
    if (a !== b) samplerSame = false;
  }
  r.ok("100 组随机 logits 上 top_p=1.0 与旧采样逐位一致", samplerSame);

  /* ---------------- 3. 滑动窗口 ---------------- */
  r.section("3. 滑动窗口掩码");
  const ctx5 = [1, 2, 3, 4, 0];
  const fwdW0 = L.llmForward(mb0, ctx5).logits;
  mb0.cfg.window = 2;
  const genW2 = L.llmGenerate(mb0, [1, 2], 12, 0.8, 0, L.llmSeedRng(7));
  const fwdW2 = L.llmForward(mb0, ctx5).logits;
  let causalOk = true;
  const fwdW2b = L.llmForward(mb0, [1, 2, 3, 4, 3]).logits;
  for (let t = 0; t < 4; t++) {
    for (let v = 0; v < 5; v++) if (fwdW2[t][v] !== fwdW2b[t][v]) causalOk = false;
  }
  const lastChanged = fwdW2[4].some((v, i) => v !== fwdW2b[4][i]);
  r.ok("window=2 生成结果与 window=0 不同", !eqArr(genW2, genW0), genW2.join(","));
  r.ok("window=2 末位 logits 与全注意力不同", !eqDeep(fwdW2[4], fwdW0[4]));
  r.ok("window=2 仍是因果的（改最后一个 token 不影响前 4 个位置）", causalOk);
  r.ok("window=2 最后一行确实变化", lastChanged);

  mb0.cfg.window = 1;
  const l1 = L.llmForward(mb0, [1, 2, 3]).logits[2];
  const l2 = L.llmForward(mb0, [4, 0, 3]).logits[2];
  r.ok("window=1 时位置 2 只看自己（换掉前两个 token 结果逐位不变）", eqDeep(l1, l2));
  mb0.cfg.window = 0;

  let badWin = 0;
  for (const w of [-1, 1.5, NaN, "abc"]) {
    try { L.llmCreate({ vocabSize: 3, dModel: 4, heads: 2, dff: 8, maxLen: 4, window: w }); }
    catch (e) { badWin++; }
  }
  r.ok("llmCreate 拒绝非法窗口（-1 / 1.5 / NaN / 字符串）", badWin === 4, "rejected=" + badWin);

  /* ---------------- 4. 中心差分梯度 ---------------- */
  r.section("4. 中心有限差分梯度");
  function gradCheck(seed, window, nPick, tag) {
    const m = L.llmCreate({ vocabSize: 4, dModel: 4, heads: 2, dff: 6, maxLen: 5, window, rand: L.llmSeedRng(seed) });
    const ids = [1, 2, 3, 0, 2];
    L.llmZeroGrad(m);
    const { cache } = L.llmForward(m, ids);
    L.llmBackward(m, cache);
    const scalars = [];
    collectScalars(m.p, m.g, "p", scalars);
    scalars.sort((a, b) => Math.abs(b.grad) - Math.abs(a.grad));
    const picked = scalars.slice(0, nPick);
    const eps = 1e-5;
    let worst = 0, worstPath = "";
    for (const s of picked) {
      const v0 = s.parent[s.key];
      s.parent[s.key] = v0 + eps;
      const lp = L.llmForward(m, ids).loss;
      s.parent[s.key] = v0 - eps;
      const lm = L.llmForward(m, ids).loss;
      s.parent[s.key] = v0;
      const num = (lp - lm) / (2 * eps);
      const rel = Math.abs(num - s.grad) / Math.max(Math.abs(num), Math.abs(s.grad), 1e-12);
      if (rel > worst) { worst = rel; worstPath = s.path; }
    }
    r.note(tag + ": " + nPick + " 个参数, 最差相对误差 " + worst.toExponential(3) +
      " (最差: " + worstPath + ")");
    return worst;
  }
  const worst0 = gradCheck(12345, 0, 10, "window=0");
  const worst2 = gradCheck(54321, 2, 10, "window=2");
  r.ok("window=0 梯度相对误差 ≤ 1e-8", worst0 <= 1e-8, worst0.toExponential(3));
  r.ok("window=2 梯度相对误差 ≤ 1e-8", worst2 <= 1e-8, worst2.toExponential(3));

  /* ---------------- 5. 困惑度 ---------------- */
  r.section("5. 困惑度");
  const pplLoss = L.llmForward(mb, idsB).loss;
  const ppl = L.llmPerplexity(pplLoss);
  r.ok("困惑度 = exp(平均交叉熵)（误差 ≤ 1e-12）",
    Math.abs(ppl - Math.exp(pplLoss)) <= 1e-12 * Math.exp(pplLoss),
    "loss=" + pplLoss + " ppl=" + ppl);
  r.ok("loss > 700 时给 inf", L.llmPerplexity(701) === Infinity);
  r.ok("loss = 0 时给 1", L.llmPerplexity(0) === 1);

  /* ---------------- 6. 存档 / 读档 ---------------- */
  r.section("6. 保存 → JSON → 加载 → 生成");
  const vocab = ["0", "1", "2", "3", "4"];
  const obj = L.llmExportModel(mb, vocab);
  const text = JSON.stringify(obj);
  r.ok("存档含 format / config / vocab / params",
    obj.format === "llm_web_v1" && obj.config && obj.vocab.length === 5 && obj.params.Wout);
  const restored = L.llmImportModel(JSON.parse(text));
  const genR1 = L.llmGenerate(mb, [1, 2], 12, 0.8, 0, L.llmSeedRng(7), 0.9);
  const genR2 = L.llmGenerate(restored.model, [1, 2], 12, 0.8, 0, L.llmSeedRng(7), 0.9);
  r.ok("读回的模型生成同样 token 序列", eqArr(genR1, genR2), genR2.join(","));
  r.ok("读回的模型 forward loss 逐位相同", L.llmForward(restored.model, idsB).loss === L.llmForward(mb, idsB).loss);
  r.ok("读回的模型参数与存档逐位相同", eqDeep(restored.model.p, JSON.parse(JSON.stringify(mb.p))));
  r.ok("读回的词表一致", eqArr(restored.vocab, vocab));
  r.ok("读回的配置一致", restored.model.cfg.window === mb.cfg.window &&
    restored.model.cfg.dModel === mb.cfg.dModel && restored.model.cfg.maxLen === mb.cfg.maxLen);

  mb0.cfg.window = 2;   /* 用非零窗口，否则「保留」和「总是写 0」无法区分 */
  const objW2 = L.llmExportModel(mb0, vocab);
  const resW2 = L.llmImportModel(JSON.parse(JSON.stringify(objW2)));
  r.ok("window 随存档保留（2 → 2）", resW2.model.cfg.window === 2);
  mb0.cfg.window = 0;

  r.note("非法文件必须给出中文错误而不是抛到面板外：");
  const badCases = [
    ["非对象", null, "模型对象"],
    ["format 不对", { format: "x" }, "format"],
    ["缺 config", { format: "llm_web_v1" }, "config"],
    ["vocabSize 与词表不符", { format: "llm_web_v1", config: { vocabSize: 3, dModel: 4, heads: 2, dff: 6, maxLen: 4 }, vocab: ["a", "b"], params: {} }, "词表长度"],
    ["重复字符", { format: "llm_web_v1", config: { vocabSize: 2, dModel: 4, heads: 2, dff: 6, maxLen: 4 }, vocab: ["a", "a"], params: {} }, "重复"],
    ["参数形状不符", { format: "llm_web_v1", config: { vocabSize: 2, dModel: 4, heads: 2, dff: 6, maxLen: 4 }, vocab: ["a", "b"], params: { E: [] } }, "参数"],
    ["参数含 null", { format: "llm_web_v1", config: { vocabSize: 2, dModel: 4, heads: 2, dff: 6, maxLen: 4 }, vocab: ["a", "b"], params: { E: [[null, 0, 0, 0], [0, 0, 0, 0]] } }, "有限数值"],
  ];
  let badOk = 0;
  for (const [name, bad, need] of badCases) {
    try {
      L.llmImportModel(bad);
      r.note("FAIL " + name + "：没有报错");
    } catch (e) {
      const hit = e instanceof Error && e.message.indexOf(need) >= 0;
      if (!hit) r.note("FAIL " + name + "：错误信息未含「" + need + "」→ " + e.message);
      else { badOk++; r.note("ok   " + name + " → " + e.message); }
    }
  }
  r.ok("7 种非法存档全部被拒绝并带清晰信息", badOk === badCases.length, badOk + "/" + badCases.length);

  /* ---- 输入校验：非法输入必须报错，而不是静默算出 NaN ---- */
  r.note("输入校验（对齐 Python 版审计 F1/F6/F7）：");
  const vm = L.llmCreate({ vocabSize: 5, dModel: 8, heads: 2, dff: 16, maxLen: 6, seed: 7, window: 0 });
  function throwsMsg(fn, needle, label) {
    try {
      fn();
      r.note("FAIL " + label + "：没有报错");
      return false;
    } catch (e) {
      const hit = e instanceof Error && e.message.indexOf(needle) >= 0;
      if (!hit) r.note("FAIL " + label + "：信息未含「" + needle + "」→ " + e.message);
      return hit;
    }
  }

  let idCase = 0;
  for (const bad of [5, -1, 1.5]) {
    if (throwsMsg(() => L.llmForward(vm, [1, 2, bad]), "token id", "非法 id " + bad)) idCase++;
  }
  r.ok("越界/负数/非整数 token id 都报错", idCase === 3, idCase + "/3");

  r.ok("整行 -Infinity 的 softmax 返回全 0 而不是 NaN",
    eqArr(L.softmaxRow([-Infinity, -Infinity, -Infinity]), [0, 0, 0]));

  const vnan = L.llmCreate({ vocabSize: 5, dModel: 8, heads: 2, dff: 16, maxLen: 6, seed: 7, window: 0 });
  vnan.p.Wout[0][0] = NaN;
  r.ok("权重含 NaN 时前向报错（不再把 NaN loss 当正常值）",
    throwsMsg(() => L.llmForward(vnan, [1, 2, 3]), "NaN", "NaN 权重"));

  r.ok("temperature<=0 报错",
    throwsMsg(() => L.llmGenerate(vm, [1], 2, 0, 0, () => 0.5, 1), "temperature", "temp=0"));
  r.ok("top_p 越界报错",
    throwsMsg(() => L.llmGenerate(vm, [1], 2, 1, 0, () => 0.5, 1.5), "top_p", "top_p=1.5") &&
    throwsMsg(() => L.llmGenerate(vm, [1], 2, 1, 0, () => 0.5, 0), "top_p", "top_p=0"));

  r.ok("合法参数照常生成（top_p 省略 = 不过滤）",
    L.llmGenerate(vm, [1], 3, 1.0, 1, () => 0.5).length === 4);

  return { checks: r.checks, failures: r.failures };
}

module.exports = { run };

if (require.main === module) {
  run().then(
    res => { process.exitCode = summarize(res.checks, res.failures); },
    err => { console.log("FAIL 套件异常: " + (err && err.stack ? err.stack : err)); process.exitCode = 1; }
  );
}
