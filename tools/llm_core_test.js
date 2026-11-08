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

  /* ---------------- 2b. 采样控制（min_p / 惩罚 / 禁重复 n-gram） ---------------- */
  r.section("2b. 采样控制（min_p / 重复 / 存在 / 频率惩罚 / n-gram）");

  r.ok("默认参数下 llmApplyPenalties 原样返回 logits",
    eqArr(L.llmApplyPenalties([2, 1, 0, -1], [0, 3]), [2, 1, 0, -1]));

  r.ok("repetition_penalty：正 logit 除以惩罚、非正 logit 乘以惩罚",
    eqArr(L.llmApplyPenalties([2, 1, 0, -1], [0, 3], { repetitionPenalty: 2 }),
      [1, 1, 0, -2]));

  r.ok("presence_penalty：每个出现过的 token 只扣一次（不管出现几次）",
    eqArr(L.llmApplyPenalties([1, 2, 3], [0, 0, 1], { presencePenalty: 0.5 }),
      [0.5, 1.5, 3]));

  r.ok("frequency_penalty：按出现次数扣",
    eqArr(L.llmApplyPenalties([1, 2, 3], [0, 0, 1], { frequencyPenalty: 0.5 }),
      [0, 1.5, 3]));

  const ng2 = L.llmApplyPenalties([0, 0, 0], [1, 2, 1, 2], { noRepeatNgramSize: 2 });
  r.ok("no_repeat_ngram_size=2：禁止末尾 1-gram 重复后的同一后继",
    ng2[0] === 0 && ng2[1] === -Infinity && ng2[2] === 0);

  const ng3 = L.llmApplyPenalties([0, 0, 0], [0, 1, 2, 0, 1], { noRepeatNgramSize: 3 });
  r.ok("no_repeat_ngram_size=3：禁止重复三元组的后继",
    ng3[0] === 0 && ng3[1] === 0 && ng3[2] === -Infinity);

  const minRng = L.llmSeedRng(9090);
  const peaked3 = [Math.log(0.7), Math.log(0.2), Math.log(0.1)];
  let onlyPeak = true;
  for (let i = 0; i < 200; i++) {
    if (L.llmSampleToken(peaked3, 1, 0, 1, minRng, 0.5) !== 0) onlyPeak = false;
  }
  r.ok("min_p=0.5 在 0.7/0.2/0.1 上只保留最大项", onlyPeak);

  let afterTopK = true;
  for (let i = 0; i < 200; i++) {
    const t = L.llmSampleToken([Math.log(0.5), Math.log(0.3), Math.log(0.2)],
      1, 2, 1, minRng, 0.9);
    if (t !== 0) afterTopK = false;
  }
  r.ok("min_p 在 top_k 之后生效（top2 再按相对概率截尾只剩最大项）", afterTopK);

  let sawBoth = [false, false];
  for (let i = 0; i < 400; i++) {
    sawBoth[L.llmSampleToken([0, 0], 1, 0, 1, minRng, 0.5)] = true;
  }
  r.ok("min_p 只按相对最大概率截尾（两个等概率 token 都保留）", sawBoth[0] && sawBoth[1]);

  const genNeutral = L.llmGenerate(mb, [1, 2], 12, 0.8, 0, L.llmSeedRng(7), 1.0, {
    minP: 0, repetitionPenalty: 1, presencePenalty: 0,
    frequencyPenalty: 0, noRepeatNgramSize: 0,
  });
  r.ok("显式全中性采样控制与基线逐位相同", eqArr(genNeutral, baseline.genA),
    genNeutral.join(","));

  const genGreedy = L.llmGenerate(mb, [1, 2], 12, 0.8, 1, L.llmSeedRng(7), 1.0);
  const genMinP = L.llmGenerate(mb, [1, 2], 12, 0.8, 2, L.llmSeedRng(7), 1.0,
    { minP: 0.999999 });
  r.ok("min_p 在续写中生效（topK=2 + min_p≈1 退化为贪心）",
    eqArr(genMinP, genGreedy), genMinP.join(","));

  const genRep = L.llmGenerate(mb, [1, 2], 12, 0.8, 0, L.llmSeedRng(7), 1.0,
    { repetitionPenalty: 4 });
  r.ok("repetition_penalty 在续写中生效（序列与基线不同）",
    !eqArr(genRep, baseline.genA), genRep.join(","));

  const genPres = L.llmGenerate(mb, [1, 2], 12, 0.8, 0, L.llmSeedRng(7), 1.0,
    { presencePenalty: 3 });
  r.ok("presence_penalty 在续写中生效（序列与基线不同）",
    !eqArr(genPres, baseline.genA), genPres.join(","));

  const genFreq = L.llmGenerate(mb, [1, 2], 12, 0.8, 0, L.llmSeedRng(7), 1.0,
    { frequencyPenalty: 3 });
  r.ok("frequency_penalty 在续写中生效（序列与基线不同）",
    !eqArr(genFreq, baseline.genA), genFreq.join(","));

  const genNg = L.llmGenerate(mb, [1, 2], 12, 0.8, 0, L.llmSeedRng(7), 1.0,
    { noRepeatNgramSize: 2 });
  const seenBigrams = new Set();
  let noRepeatBigram = true;
  for (let i = 1; i < genNg.length; i++) {
    const key = genNg[i - 1] + "," + genNg[i];
    if (seenBigrams.has(key)) noRepeatBigram = false;
    seenBigrams.add(key);
  }
  r.ok("no_repeat_ngram_size=2 在续写中生效（结果没有重复二元组）",
    noRepeatBigram && !eqArr(genNg, baseline.genA), genNg.join(","));

  const badGenOpts = [
    ["min_p=-0.1", { minP: -0.1 }, "min_p"],
    ["min_p=1", { minP: 1 }, "min_p"],
    ["min_p=NaN", { minP: NaN }, "min_p"],
    ["min_p 字符串", { minP: "0.5" }, "min_p"],
    ["ngram=1", { noRepeatNgramSize: 1 }, "no_repeat_ngram_size"],
    ["ngram=1.5", { noRepeatNgramSize: 1.5 }, "no_repeat_ngram_size"],
    ["repetition_penalty=0", { repetitionPenalty: 0 }, "repetition_penalty"],
    ["presence_penalty 字符串", { presencePenalty: "x" }, "presence_penalty"],
  ];
  let badGenOk = 0;
  for (const [label, opt, needle] of badGenOpts) {
    try {
      L.llmGenerate(mb, [1], 2, 1, 0, () => 0.5, 1, opt);
      r.note("FAIL " + label + "：没有报错");
    } catch (e) {
      const hit = e instanceof Error && e.message.indexOf(needle) >= 0;
      if (!hit) r.note("FAIL " + label + "：信息未含「" + needle + "」→ " + e.message);
      else badGenOk++;
    }
  }
  r.ok("8 种非法生成参数全部被拒绝", badGenOk === badGenOpts.length,
    badGenOk + "/" + badGenOpts.length);

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


  r.section("W7. typical / top-a / eta / tail-free / mirostat");

  /* 黄金值来自 Python 侧实测：AI.typical_filter 等对同一行 logits 的保留集合，
     行 = [0.5, 2.0, -1.0, 1.0, 3.0, 0.0, -0.5, 1.5]（V = 8）。 */
  const w7row = [0.5, 2.0, -1.0, 1.0, 3.0, 0.0, -0.5, 1.5];
  const keptIdx = row => {
    const out = [];
    for (let i = 0; i < row.length; i++) if (Number.isFinite(row[i])) out.push(i);
    return out;
  };

  r.ok("typical(mass=0.9) 保留 {1,3,4,7}",
    eqArr(keptIdx(L.llmMaskTypical(w7row, 0.9)), [1, 3, 4, 7]),
    JSON.stringify(keptIdx(L.llmMaskTypical(w7row, 0.9))));
  r.ok("typical 默认 mass=0.9",
    eqArr(keptIdx(L.llmMaskTypical(w7row)), [1, 3, 4, 7]));

  r.ok("top_a(0.1) 保留 {0,1,3,4,7}",
    eqArr(keptIdx(L.llmMaskTopA(w7row, 0.1)), [0, 1, 3, 4, 7]),
    JSON.stringify(keptIdx(L.llmMaskTopA(w7row, 0.1))));
  r.ok("top_a=0 不过滤（原样返回）",
    eqArr(L.llmMaskTopA(w7row, 0), w7row));

  r.ok("eta(0.3) 只留 argmax {4}",
    eqArr(keptIdx(L.llmMaskEta(w7row, 0.3)), [4]),
    JSON.stringify(keptIdx(L.llmMaskEta(w7row, 0.3))));
  r.ok("eta 至少保留 argmax（全等概率时留 0）",
    eqArr(keptIdx(L.llmMaskEta([0, 0, 0], 0.9)), [0]));

  r.ok("tail_free(0.5) 保留 {1,4,7}",
    eqArr(keptIdx(L.llmMaskTailFree(w7row, 0.5)), [1, 4, 7]),
    JSON.stringify(keptIdx(L.llmMaskTailFree(w7row, 0.5))));
  r.ok("tail_free 在 n<=2 时原样返回",
    eqArr(L.llmMaskTailFree([0, 0], 0.5), [0, 0]));

  const mir = L.llmMirostatFilter(w7row, 2.0);
  r.ok("mirostat(mu=2) 保留 {1,4}",
    eqArr(keptIdx(mir.logits), [1, 4]), JSON.stringify(keptIdx(mir.logits)));
  r.ok("mirostat 保留集熵 = 0.582203108888218",
    Math.abs(mir.entropy - 0.582203108888218) < 1e-15,
    String(mir.entropy));

  r.ok("mirostat_update(5, obs=1, target=2, 0.2) = 5.2",
    L.llmMirostatUpdate(5, 1, 2, 0.2) === 5.2, String(L.llmMirostatUpdate(5, 1, 2, 0.2)));
  r.ok("mirostat_update 下限夹到 0",
    L.llmMirostatUpdate(0.1, 5, 2, 0.2) === 0, String(L.llmMirostatUpdate(0.1, 5, 2, 0.2)));

  r.ok("typical/top_a/eta/tft/mirostat 的参数越界都报错",
    throwsMsg(() => L.llmMaskTypical(w7row, 0), "mass", "typical=0") &&
    throwsMsg(() => L.llmMaskTopA(w7row, 1.5), "top_a", "top_a=1.5") &&
    throwsMsg(() => L.llmMaskEta(w7row, 1), "eta", "eta=1") &&
    throwsMsg(() => L.llmMaskTailFree(w7row, 0), "tft", "tft=0") &&
    throwsMsg(() => L.llmMirostatFilter(w7row, -1), "mu", "mu=-1"));

  /* 关掉过滤器时与旧签名逐位一致：同一 rand 序列应得到同一串 token。 */
  {
    const a = L.llmGenerate(vm, [1], 4, 1.0, 2, () => 0.42);
    const b = L.llmGenerate(vm, [1], 4, 1.0, 2, () => 0.42, 1.0, {});
    r.ok("filters 为空对象时与旧路径逐位一致", eqArr(a, b), JSON.stringify([a, b]));
  }
  {
    const c = L.llmGenerate(vm, [1], 3, 1.0, 0, () => 0.42, 1.0, { typical: 0.9 });
    r.ok("typical 选项可驱动生成", c.length === 4, JSON.stringify(c));
  }
  r.ok("非法 typical/tft/mirostat_mu 在 generate 入口就报错",
    throwsMsg(() => L.llmGenerate(vm, [1], 2, 1, 0, () => 0.5, 1, { typical: 2 }), "typical", "typical=2") &&
    throwsMsg(() => L.llmGenerate(vm, [1], 2, 1, 0, () => 0.5, 1, { tft: 0 }), "tft", "tft=0") &&
    throwsMsg(() => L.llmGenerate(vm, [1], 2, 1, 0, () => 0.5, 1, { mirostatMu: -1 }), "mirostat", "mu=-1"));

  testJsonGrammar(L, r);

  testPrefixCache(L, r);

  testContrastive(L, r);

  testRegexGrammar(L, r);

  testCfgGrammar(L, r);

  testCharJsonGrammar(L, r);

  testIncrementalForward(L, r);

  return { checks: r.checks, failures: r.failures };
}

module.exports = { run };

if (require.main === module) {
  run().then(
    res => { process.exitCode = summarize(res.checks, res.failures); },
    err => { console.log("FAIL 套件异常: " + (err && err.stack ? err.stack : err)); process.exitCode = 1; }
  );
}

/* 词表级 JSON 约束（Python grammar_allowed 的等价状态机） */
function testJsonGrammar(L, r) {
  const VOCAB = ["{", "}", "[", "]", ":", ",",
    '"a"', '"b"', "1", "-2.5", "0", "true", "false", "null", "x"];
  const CASES = [
    { p: [], a: ['"a"', '"b"', "-2.5", "0", "1", "[", "false", "null", "true", "{"] },
    { p: ["{"], a: ['"a"', '"b"', "}"] },
    { p: ["{", '"a"'], a: [":"] },
    { p: ["{", '"a"', ":"], a: ['"a"', '"b"', "-2.5", "0", "1", "[", "false", "null", "true", "{"] },
    { p: ["{", '"a"', ":", "1"], a: [",", "}"] },
    { p: ["{", '"a"', ":", "1", ","], a: ['"a"', '"b"'] },
    { p: ["{", '"a"', ":", "1", "}"], a: [] },
    { p: ["["], a: ['"a"', '"b"', "-2.5", "0", "1", "[", "]", "false", "null", "true", "{"] },
    { p: ["[", "1"], a: [",", "]"] },
    { p: ["[", "1", ","], a: ['"a"', '"b"', "-2.5", "0", "1", "[", "false", "null", "true", "{"] },
    { p: ["[", "1", "]"], a: [] },
    { p: ["[", "["], a: ['"a"', '"b"', "-2.5", "0", "1", "[", "]", "false", "null", "true", "{"] },
    { p: ["{", '"a"', ":", "[", "true"], a: [",", "]"] },
    { p: ["1"], a: [] },
    { p: ["true"], a: [] },
  ];

  r.section("7. 词表级 JSON 约束解码（等价于 Python json_grammar）");

  for (const c of CASES) {
    let ctx = L.llmJsonStart();
    let walked = true;
    for (const tok of c.p) {
      ctx = L.llmJsonStep(ctx, tok);
      if (ctx === null) { walked = false; break; }
    }
    r.ok("前缀可走通 " + JSON.stringify(c.p), walked, "");
    if (!walked) continue;
    const got = L.llmJsonAllowed(ctx, VOCAB).slice().sort();
    const want = c.a.slice().sort();
    r.ok("允许集合 " + JSON.stringify(c.p),
      JSON.stringify(got) === JSON.stringify(want), JSON.stringify(got) + " vs " + JSON.stringify(want));
    const mask = L.llmJsonMask(ctx, VOCAB);
    let maskOk = mask.length === VOCAB.length;
    for (let k = 0; k < VOCAB.length; k++) {
      const inAllowed = c.a.indexOf(VOCAB[k]) >= 0;
      if (mask[k] !== inAllowed) maskOk = false;
    }
    r.ok("掩码与集合一致 " + JSON.stringify(c.p), maskOk, JSON.stringify(mask));
    r.ok("完成判定 " + JSON.stringify(c.p),
      L.llmJsonComplete(ctx) === (c.a.length === 0), String(L.llmJsonComplete(ctx)));
  }

  const s0 = L.llmJsonStart();
  r.ok("未知 token 返回 null", L.llmJsonStep(s0, "xyz") === null, "");
  r.ok("顶层不能直接给 }", L.llmJsonStep(s0, "}") === null, "");
  r.ok("键不能是数字", L.llmJsonStep({ st: "KEY", stack: [] }, "1") === null, "");
  r.ok("token 必须字符串", L.llmJsonStep(s0, 42) === null, "");
  r.ok("非法 ctx 抛错", (function () { try { L.llmJsonStep(null, "{"); return false; } catch (e) { return true; } })(), "");

  let ctx = L.llmJsonStart();
  const doc = ["{", '"a"', ":", "[", "1", ",", "true", "]", "}"];
  let ok = true;
  for (const tok of doc) {
    ctx = L.llmJsonStep(ctx, tok);
    if (ctx === null) { ok = false; break; }
  }
  r.ok("完整文档走到 DONE", ok && L.llmJsonComplete(ctx), JSON.stringify(ctx));
  r.ok("DONE 后拒绝新 token", L.llmJsonStep(ctx, "}") === null, "");

  const lp = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02];
  r.ok("pick 跳过非法高分 token", VOCAB[L.llmJsonPick(lp, VOCAB, L.llmJsonStart())] === "{",
    VOCAB[L.llmJsonPick(lp, VOCAB, L.llmJsonStart())]);
  const lk = [0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  r.ok("pick 在 KCLOSE 选字符串键", VOCAB[L.llmJsonPick(lk, VOCAB, { st: "KCLOSE", stack: [] })] === '"b"',
    VOCAB[L.llmJsonPick(lk, VOCAB, { st: "KCLOSE", stack: [] })]);

  let threw = 0;
  try { L.llmJsonPick([1.0], VOCAB, s0); } catch (e) { threw++; }
  try { L.llmJsonPick([1.0], ["}"], s0); } catch (e) { threw++; }
  try { L.llmJsonPick([1.0], [{}], s0); } catch (e) { threw++; }
  r.ok("pick 三类错误都抛", threw === 3, String(threw));

  let violations = 0;
  const rnd = L.llmSeedRng(20261108);
  for (let trial = 0; trial < 300; trial++) {
    let c = L.llmJsonStart();
    for (let step = 0; step < 8; step++) {
      const allowed = L.llmJsonAllowed(c, VOCAB);
      if (!allowed.length) break;
      const logits = VOCAB.map(function () { return L.llmGauss(rnd); });
      const tok = VOCAB[L.llmJsonPick(logits, VOCAB, c)];
      if (allowed.indexOf(tok) < 0) violations++;
      c = L.llmJsonStep(c, tok);
      if (c === null) { violations++; break; }
    }
  }
  r.ok("随机行走 300 次无非法 token", violations === 0, String(violations));
}

/* 前缀 KV 缓存（Python PrefixCache 的 JS 版） */
function testPrefixCache(L, r) {
  r.section("8. 前缀 KV 缓存（PrefixCache）");

  const c = L.llmPrefixCache(2, 3);
  r.ok("新建缓存每层为空", L.llmCacheLength(c, 0) === 0 && L.llmCacheLength(c, 1) === 0, "");

  L.llmCacheSet(c, 0, [[1, 2, 3], [4, 5, 6]], [[0.5, 0.5, 0.5], [0.1, 0.2, 0.3]]);
  r.ok("set 后长度", L.llmCacheLength(c, 0) === 2, String(L.llmCacheLength(c, 0)));
  const got = L.llmCacheGet(c, 0);
  r.ok("get 往返", JSON.stringify(got.keys) === "[[1,2,3],[4,5,6]]", JSON.stringify(got.keys));
  got.keys[0][0] = 99;
  r.ok("get 返回深拷贝", L.llmCacheGet(c, 0).keys[0][0] === 1, String(L.llmCacheGet(c, 0).keys[0][0]));

  r.ok("truncate 截到 1", (L.llmCacheTruncate(c, 1), L.llmCacheLength(c, 0) === 1), String(L.llmCacheLength(c, 0)));
  r.ok("truncate 不放大", (L.llmCacheTruncate(c, 5), L.llmCacheLength(c, 0) === 1), String(L.llmCacheLength(c, 0)));

  const other = L.llmPrefixCache(2, 3);
  L.llmCacheSet(other, 0, [[7, 8, 9]], [[1, 1, 1]]);
  L.llmCacheMerge(c, other);
  r.ok("merge 追加", L.llmCacheLength(c, 0) === 2, String(L.llmCacheLength(c, 0)));
  r.ok("merge 内容正确", JSON.stringify(L.llmCacheGet(c, 0).keys) === "[[1,2,3],[7,8,9]]",
    JSON.stringify(L.llmCacheGet(c, 0).keys));
  r.ok("未 set 的层 merge 后仍为空", L.llmCacheLength(c, 1) === 0, String(L.llmCacheLength(c, 1)));

  const json = JSON.parse(JSON.stringify(L.llmCacheToJSON(c)));
  const back = L.llmCacheFromJSON(json);
  r.ok("JSON 往返", JSON.stringify(L.llmCacheToJSON(back)) === JSON.stringify(L.llmCacheToJSON(c)),
    JSON.stringify(L.llmCacheToJSON(back)));
  r.ok("fromJSON 结果可继续 merge", (L.llmCacheMerge(back, other), L.llmCacheLength(back, 0) === 3),
    String(L.llmCacheLength(back, 0)));

  const stats = L.llmCacheStats(c, 4);
  r.ok("stats 复用率", Math.abs(stats[0].reuse - 0.5) < 1e-12 && Math.abs(stats[1].reuse) < 1e-12,
    JSON.stringify(stats));
  r.ok("promptLen=0 时复用率为 0", L.llmCacheStats(c, 0)[0].reuse === 0, "");
  const bars = L.llmCacheBars(stats, 10);
  r.ok("bars 每层一行且用 #/.", bars.length === 2 && /^L0 #{5}\.{5} len=2 reuse=0\.500$/.test(bars[0]),
    bars[0]);

  /* 错误路径 */
  let threw = 0;
  const bad = function (fn) { try { fn(); } catch (e) { threw++; } };
  bad(function () { L.llmPrefixCache(0, 3); });
  bad(function () { L.llmPrefixCache(2, 0); });
  bad(function () { L.llmCacheSet(c, 5, [], []); });
  bad(function () { L.llmCacheSet(c, 0, [[1, 2]], [[1, 2, 3]]); });
  bad(function () { L.llmCacheSet(c, 0, [[1, NaN, 3]], [[1, 2, 3]]); });
  bad(function () { L.llmCacheSet(c, 0, [[1, 2, 3]], [[1, 2, 3], [1, 2, 3]]); });
  bad(function () { L.llmCacheTruncate(c, -1); });
  bad(function () { L.llmCacheMerge(c, L.llmPrefixCache(3, 3)); });
  bad(function () { L.llmCacheMerge(c, L.llmPrefixCache(2, 4)); });
  bad(function () { L.llmCacheFromJSON({ numLayers: 2, dModel: 3, keys: [], values: [] }); });
  bad(function () { L.llmCacheFromJSON({ numLayers: 2, dModel: 3, keys: [[], []], values: [] }); });
  bad(function () { L.llmCacheStats(c, -1); });
  r.ok("12 条非法输入都报错", threw === 12, String(threw));

  /* 与 Python PrefixCache 的语义对照：merge 等价于「先 set 再追加」 */
  const py = L.llmPrefixCache(1, 2);
  L.llmCacheSet(py, 0, [[1, 2]], [[3, 4]]);
  const tail = L.llmPrefixCache(1, 2);
  L.llmCacheSet(tail, 0, [[5, 6]], [[7, 8]]);
  L.llmCacheMerge(py, tail);
  r.ok("两次追加与一次 set 等价",
    JSON.stringify(L.llmCacheGet(py, 0).keys) === "[[1,2],[5,6]]" &&
    JSON.stringify(L.llmCacheGet(py, 0).values) === "[[3,4],[7,8]]",
    JSON.stringify(L.llmCacheGet(py, 0)));
}

/* ---------------- 新增测试用的小工具 ---------------- */

function tThrowsMsg(fn, needle) {
  try {
    fn();
    return false;
  } catch (e) {
    return e instanceof Error && e.message.indexOf(needle) >= 0;
  }
}

function tEqBits(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/* 对比搜索（Python contrastive_select / contrastive_search 的 JS 版） */
function testContrastive(L, r) {
  r.section("9. 对比搜索（contrastive search）");

  /* 打分：score = (1−α)·p − α·reward；reward = 候选与历史隐状态的最大余弦 */
  const lg = [1.0, 1.0, 1.0];
  const hist = [[1, 0]];
  const hiddens = { 0: [1, 0], 1: [0, 1], 2: [0, 1] };
  const pick1 = L.llmContrastiveSelect(lg, hiddens, hist, 0.5, 2);
  r.ok("top-2 候选是 {0,1}", eqArr(pick1.candidates.map(function (c) { return c.token; }), [0, 1]),
    JSON.stringify(pick1.candidates.map(function (c) { return c.token; })));
  const c0 = pick1.candidates[0], c1 = pick1.candidates[1];
  r.ok("候选 0 与历史同向 → reward=1", Math.abs(c0.reward - 1) < 1e-15, String(c0.reward));
  r.ok("候选 1 与历史正交 → reward=0", Math.abs(c1.reward) < 1e-15, String(c1.reward));
  r.ok("score 公式 (1−α)p − α·reward",
    Math.abs(c0.score - (0.5 * (1 / 3) - 0.5 * 1)) < 1e-15 &&
    Math.abs(c1.score - (0.5 * (1 / 3) - 0.5 * 0)) < 1e-15,
    JSON.stringify([c0.score, c1.score]));
  r.ok("alpha=0.5 时挑低相似度候选", pick1.token === 1, String(pick1.token));

  const pick2 = L.llmContrastiveSelect(lg, hiddens, hist, 0, 2);
  r.ok("alpha=0 退化为按概率（同分保留较小 token）", pick2.token === 0, String(pick2.token));
  const pick3 = L.llmContrastiveSelect(lg, hiddens, hist, 1, 2);
  r.ok("alpha=1 只看相似度（挑 reward 更小的）", pick3.token === 1, String(pick3.token));

  const pickK1 = L.llmContrastiveSelect([0.1, 0.2, 0.3], hiddens, [], 0.6, 1);
  r.ok("top_k=1 只保留概率最高的候选", pickK1.token === 2 && pickK1.candidates.length === 1,
    JSON.stringify(pickK1));

  const z = L.llmContrastiveSelect([1, 1], { 0: [0, 0], 1: [1, 0] }, [[1, 0]], 0.5, 2);
  r.ok("零向量隐状态 reward=0（不除零）且不被惩罚选中",
    z.token === 0 && Math.abs(z.candidates[0].reward) < 1e-15, JSON.stringify(z));

  r.ok("alpha 越界 / top_k 非正 / hiddens 空 / 缺候选 / 宽度不一致 都报错",
    tThrowsMsg(function () { L.llmContrastiveSelect(lg, hiddens, hist, 1.5, 2); }, "alpha") &&
    tThrowsMsg(function () { L.llmContrastiveSelect(lg, hiddens, hist, -0.1, 2); }, "alpha") &&
    tThrowsMsg(function () { L.llmContrastiveSelect(lg, hiddens, hist, 0.5, 0); }, "top_k") &&
    tThrowsMsg(function () { L.llmContrastiveSelect(lg, {}, hist, 0.5, 2); }, "hiddens") &&
    tThrowsMsg(function () { L.llmContrastiveSelect([1, 1, 1, 1], { 0: [1, 0] }, hist, 0.5, 2); }, "缺少候选") &&
    tThrowsMsg(function () { L.llmContrastiveSelect(lg, { 0: [1, 0, 0], 1: [0, 1], 2: [0, 1] }, hist, 0.5, 2); }, "宽度"));

  /* 生成：top_k=1 等价贪心；top_k=2 每步前向 top_k+1 次 */
  const cm = L.llmCreate({ vocabSize: 5, dModel: 4, heads: 2, dff: 6, maxLen: 8, rand: L.llmSeedRng(777) });
  const ids0 = [1, 2];
  const cs1 = L.llmContrastiveSearch(cm, ids0, 3, 1, 0.6);
  r.ok("top_k=1 返回 3 个新 token 与 3 条 picks",
    cs1.ids.length === 5 && cs1.picks.length === 3, JSON.stringify(cs1.ids));

  let ctx = ids0.slice(), greedyOk = true;
  for (let s = 0; s < 3; s++) {
    const row = L.llmForward(cm, ctx).logits;
    const lastRow = row[row.length - 1];
    let best = 0;
    for (let i = 1; i < lastRow.length; i++) if (lastRow[i] > lastRow[best]) best = i;
    if (cs1.ids[s + ids0.length] !== best) { greedyOk = false; break; }
    ctx.push(best);
  }
  r.ok("top_k=1 逐步等于 argmax（与整段前向复核）", greedyOk, JSON.stringify(cs1.ids));
  r.ok("前向次数 = 1 + 步数 × top_k", cs1.forwards === 4, String(cs1.forwards));

  const cs2 = L.llmContrastiveSearch(cm, ids0, 2, 2, 0.6);
  r.ok("picks 带 score/prob/reward 且 token 与 ids 对应",
    cs2.picks.every(function (pk, i) {
      return pk.token === cs2.ids[ids0.length + i] &&
        typeof pk.score === "number" && typeof pk.prob === "number" && typeof pk.reward === "number";
    }),
    JSON.stringify(cs2.picks));
  r.ok("top_k=2 时前向次数 = 1 + 2×2", cs2.forwards === 5, String(cs2.forwards));
  r.ok("reward 是余弦相似度，落在 [-1,1]", cs2.picks.every(function (pk) { return pk.reward >= -1 && pk.reward <= 1; }),
    JSON.stringify(cs2.picks.map(function (pk) { return pk.reward; })));

  r.ok("生成入口的错误路径都报错",
    tThrowsMsg(function () { L.llmContrastiveSearch(cm, ids0, 2, 2, 1.5); }, "alpha") &&
    tThrowsMsg(function () { L.llmContrastiveSearch(cm, ids0, 0, 1, 0.6); }, "max_new_tokens") &&
    tThrowsMsg(function () { L.llmContrastiveSearch(cm, ids0, 2, 0, 0.6); }, "top_k") &&
    tThrowsMsg(function () { L.llmContrastiveSearch(cm, [], 2, 1, 0.6); }, "ids") &&
    tThrowsMsg(function () { L.llmContrastiveSearch(cm, [1, 2, 3, 4, 5, 6, 7], 3, 1, 0.6); }, "maxLen") &&
    tThrowsMsg(function () { L.llmContrastiveSearch(cm, [1, 2], 7, 1, 0.6); }, "maxLen"));
}

/* 正则约束解码（Thompson NFA，整段匹配，ASCII 字符类语义） */
function testRegexGrammar(L, r) {
  r.section("10. 正则约束解码（Thompson NFA）");

  const g = L.llmRegexGrammar("a[bc]+d");
  r.ok("匹配 abd", L.llmRegexAccepts(g, ["a", "b", "d"]));
  r.ok("匹配 abcd", L.llmRegexAccepts(g, ["a", "b", "c", "d"]));
  r.ok("拒绝 ad（+ 至少吃一个）", !L.llmRegexAccepts(g, ["a", "d"]));
  r.ok("拒绝 ab（还没有收尾 d）", !L.llmRegexAccepts(g, ["a", "b"]));
  r.ok("拒绝多字符 token（按单字符推进）", !L.llmRegexAccepts(g, ["ab"]));
  r.ok("拒绝多吃的 abd x", !L.llmRegexAccepts(g, ["a", "b", "d", "x"]));

  r.ok("空前缀可行", L.llmRegexViable(g, []));
  r.ok("非法前缀不可行", !L.llmRegexViable(g, ["x"]));
  r.ok("收尾后仍视为可行（已经能匹配）", L.llmRegexViable(g, ["a", "b", "d"]));

  const vocab = L.llmRegexChoices();
  r.ok("choices 是 32..126 的 95 个字符", vocab.length === 95 && vocab[0] === " " && vocab[94] === "~",
    vocab.length + " 个");
  r.ok("空前缀只允许 a",
    tEqBits(L.llmRegexAllowed(g, [], vocab), ["a"]), JSON.stringify(L.llmRegexAllowed(g, [], vocab)));
  r.ok("a 之后允许 b/c",
    tEqBits(L.llmRegexAllowed(g, ["a"], vocab).slice().sort(), ["b", "c"]),
    JSON.stringify(L.llmRegexAllowed(g, ["a"], vocab)));
  r.ok("ab 之后允许 b/c/d",
    tEqBits(L.llmRegexAllowed(g, ["a", "b"], vocab).slice().sort(), ["b", "c", "d"]),
    JSON.stringify(L.llmRegexAllowed(g, ["a", "b"], vocab)));
  r.ok("收尾后没有可接字符", L.llmRegexAllowed(g, ["a", "b", "d"], vocab).length === 0);
  r.ok("不可行前缀允许集为空", L.llmRegexAllowed(g, ["x"], vocab).length === 0);

  const mask = L.llmGrammarMask(g, [], vocab);
  r.ok("mask 与 allowed 一致",
    mask.length === vocab.length && mask[vocab.indexOf("a")] === true &&
    mask[vocab.indexOf("b")] === false, JSON.stringify(mask.length));
  const pickLogits = vocab.map(function (c) { return c === "b" ? 9 : (c === "a" ? 1 : 0); });
  const pk = L.llmGrammarPick(g, [], pickLogits, vocab);
  r.ok("pick 跳过不允许的高分字符（b 不可用则选 a）", pk.token === "a", pk.token);
  r.ok("pick 无解时报错",
    tThrowsMsg(function () { L.llmGrammarPick(g, ["a", "b", "d"], vocab.map(function () { return 0; }), vocab); },
      "没有可用"));

  const gd = L.llmRegexGrammar("\\d+");
  r.ok("\\d+ 匹配 005", L.llmRegexAccepts(gd, Array.from("005")));
  r.ok("\\d+ 拒绝字母", !L.llmRegexAccepts(gd, Array.from("a5")));
  r.ok("\\d 是 ASCII 数字（阿拉伯数字 ٣ 不算）", !L.llmRegexAccepts(gd, ["٣"]));

  const gw = L.llmRegexGrammar("\\w\\d");
  r.ok("\\w 是 ASCII 字母数字（下划线不算）",
    L.llmRegexAccepts(gw, ["a", "5"]) && L.llmRegexAccepts(gw, ["Z", "0"]) &&
    !L.llmRegexAccepts(gw, ["_", "5"]));

  const ga = L.llmRegexGrammar("(ab|cd)e");
  r.ok("分组 + 分支", L.llmRegexAccepts(ga, ["a", "b", "e"]) && L.llmRegexAccepts(ga, ["c", "d", "e"]) &&
    !L.llmRegexAccepts(ga, ["a", "d", "e"]));

  const gr = L.llmRegexGrammar("[a-cx]?z");
  r.ok("字符类区间与可选", L.llmRegexAccepts(gr, ["z"]) && L.llmRegexAccepts(gr, ["b", "z"]) &&
    L.llmRegexAccepts(gr, ["x", "z"]) && !L.llmRegexAccepts(gr, ["d", "z"]));

  const gn = L.llmRegexGrammar("[^a]");
  r.ok("否定字符类", L.llmRegexAccepts(gn, ["b"]) && !L.llmRegexAccepts(gn, ["a"]) &&
    !L.llmRegexAccepts(gn, ["ab"]));

  r.ok("正则解析错误都报错",
    tThrowsMsg(function () { L.llmRegexGrammar("a("); }, "分组") &&
    tThrowsMsg(function () { L.llmRegexGrammar("[z-a]"); }, "区间") &&
    tThrowsMsg(function () { L.llmRegexGrammar(5); }, "pattern") &&
    tThrowsMsg(function () { L.llmRegexGrammar("a["); }, "没有闭合") &&
    tThrowsMsg(function () { L.llmRegexState(g, ["x"]); }, "不可能匹配"));
}

/* CFG（Earley）约束解码 */
function testCfgGrammar(L, r) {
  r.section("11. CFG（Earley）约束解码");

  const g = L.llmCfgGrammar({ S: [["a", "S", "b"], ["a", "b"]] }, "S");
  r.ok("匹配 ab", L.llmCfgAccepts(g, ["a", "b"]));
  r.ok("匹配 aabb", L.llmCfgAccepts(g, ["a", "a", "b", "b"]));
  r.ok("匹配 aaabbb", L.llmCfgAccepts(g, ["a", "a", "a", "b", "b", "b"]));
  r.ok("拒绝 aab", !L.llmCfgAccepts(g, ["a", "a", "b"]));
  r.ok("拒绝 b", !L.llmCfgAccepts(g, ["b"]));
  r.ok("空串不匹配", !L.llmCfgAccepts(g, []));
  r.ok("accepts 也接受字符串输入", L.llmCfgAccepts(g, "ab"));

  r.ok("expected 空前缀 = [a]",
    tEqBits(L.llmCfgExpected(g, []), ["a"]), JSON.stringify(L.llmCfgExpected(g, [])));
  r.ok("expected 在 a 之后 = [a,b]",
    tEqBits(L.llmCfgExpected(g, ["a"]).slice().sort(), ["a", "b"]),
    JSON.stringify(L.llmCfgExpected(g, ["a"])));
  r.ok("allowed 在 a 之后可选 a/b",
    tEqBits(L.llmCfgAllowed(g, ["a"], ["a", "b", "c"]).slice().sort(), ["a", "b"]),
    JSON.stringify(L.llmCfgAllowed(g, ["a"], ["a", "b", "c"])));
  r.ok("allowed 在完整 ab 之后为空（无法再延长）",
    tEqBits(L.llmCfgAllowed(g, ["a", "b"], ["a", "b"]), []),
    JSON.stringify(L.llmCfgAllowed(g, ["a", "b"], ["a", "b"])));

  /* 谓词终结符 + 空产生式 */
  const gp = L.llmCfgGrammar({ N: [[function (ch) { return /[0-9]/.test(ch); }, "N"], []] }, "N");
  r.ok("谓词终结符匹配数字串", L.llmCfgAccepts(gp, ["1", "2", "3"]));
  r.ok("空产生式匹配空串", L.llmCfgAccepts(gp, []));
  r.ok("谓词终结符拒绝非数字", !L.llmCfgAccepts(gp, ["1", "x"]));

  r.ok("CFG 错误路径都报错",
    tThrowsMsg(function () { L.llmCfgGrammar({}, "S"); }, "起始符号") &&
    tThrowsMsg(function () { L.llmCfgGrammar({ A: [["x"]] }, "S"); }, "起始符号") &&
    tThrowsMsg(function () { L.llmCfgGrammar({ S: "x" }, "S"); }, "右部") &&
    tThrowsMsg(function () { L.llmCfgChart({ kind: "regex" }, []); }, "llmCfgGrammar") &&
    tThrowsMsg(function () { L.llmCfgAllowed(g, [], "abc"); }, "vocab"));
}

/* 字符级完整 JSON 文法（结构 + 转义 + 数字；不含 JSON Schema 校验） */
function testCharJsonGrammar(L, r) {
  r.section("12. 字符级 JSON 文法约束");

  const g = L.llmJsonCharGrammar();
  r.ok("accepts {\"a\":1}", L.llmGrammarAccepts(g, Array.from('{"a":1}')));
  r.ok("accepts {}", L.llmGrammarAccepts(g, Array.from("{}")));
  r.ok("accepts []", L.llmGrammarAccepts(g, Array.from("[]")));
  r.ok("accepts 嵌套值", L.llmGrammarAccepts(g, Array.from('[{"a":[]},true,null,-2.5e+3]')));
  r.ok("accepts 带转义的字符串",
    L.llmGrammarAccepts(g, ['"', "a", "\\", "u", "0", "0", "4", "1", '"']));
  r.ok("accepts 简单字符串", L.llmGrammarAccepts(g, Array.from('"hello"')));

  r.ok("拒绝尾逗号", !L.llmGrammarAccepts(g, Array.from('{"a":1,}')));
  r.ok("拒绝裸键", !L.llmGrammarAccepts(g, Array.from("{a:1}")));
  r.ok("拒绝单引号", !L.llmGrammarAccepts(g, Array.from("{'a':1}")));
  r.ok("拒绝前导零 01", !L.llmGrammarAccepts(g, Array.from("01")));
  r.ok("拒绝单独的 -", !L.llmGrammarAccepts(g, ["-"]));
  r.ok("拒绝 .5", !L.llmGrammarAccepts(g, Array.from(".5")));
  r.ok("拒绝 1.", !L.llmGrammarAccepts(g, Array.from("1.")));
  r.ok("拒绝 1e", !L.llmGrammarAccepts(g, Array.from("1e")));
  r.ok("拒绝 JSON 文本里的空白", !L.llmGrammarAccepts(g, Array.from(' {"a":1}')));
  r.ok("拒绝控制字符直接进字符串",
    !L.llmGrammarAccepts(g, ['"', String.fromCharCode(1), '"']));
  r.ok("拒绝非法转义 \\q", !L.llmGrammarAccepts(g, ['"', "\\", "q", '"']));

  const jv = ["{", "}", "[", "]", '"', ":", ",", "0", "1", "-", ".", "t", "r", "u", "e", "f", "a", "l", "s", "n"];
  const startAllowed = L.llmGrammarAllowed(g, [], jv);
  ["{", "[", '"', "-", "0", "1", "t", "f", "n"].forEach(function (c) {
    r.ok("空前缀允许 " + JSON.stringify(c), startAllowed.indexOf(c) >= 0, JSON.stringify(startAllowed));
  });
  r.ok("空前缀不允许 }", startAllowed.indexOf("}") < 0, JSON.stringify(startAllowed));
  r.ok("左花括号后允许键字符串或 }",
    tEqBits(L.llmGrammarAllowed(g, ["{"], jv).slice().sort(), ['"', "}"]),
    JSON.stringify(L.llmGrammarAllowed(g, ["{"], jv)));
  const valueStart = L.llmGrammarAllowed(g, ["{", '"', "a", '"', ":"], jv);
  ["{", "[", '"', "-", "0", "t", "f", "n"].forEach(function (c) {
    r.ok("值起始位置允许 " + JSON.stringify(c), valueStart.indexOf(c) >= 0, JSON.stringify(valueStart));
  });

  /* 贪心生成：零参数模型（logits 全 0）→ 每个位置取词表里最低下标的允许字符。
     词表以 { } 开头，所以会稳定生成 "{}" 并在文法接受后停止。 */
  const chars = Array.from('{}[]:",-+.0123456789truefalsn');
  const mz = L.llmCreate({ vocabSize: chars.length, dModel: 4, heads: 2, dff: 6, maxLen: 24, rand: L.llmSeedRng(9) });
  Object.keys(mz.p).forEach(function (key) {
    const W = mz.p[key];
    if (Array.isArray(W) && Array.isArray(W[0])) {
      W.forEach(function (row) { for (let i = 0; i < row.length; i++) row[i] = 0; });
    } else if (Array.isArray(W)) {
      for (let i = 0; i < W.length; i++) W[i] = 0;
    }
  });
  const gen = L.llmGrammarGenerate(mz, [chars.indexOf("{")], chars, g, 16);
  r.ok("贪心约束生成产出合法 JSON 并收尾",
    gen.complete && JSON.stringify(gen.tokens) === '["{","}"]', JSON.stringify(gen.tokens));

  /* 采样版：正则在 "ab" 上每一步都只有一个允许字符，生成结果确定 */
  const chars4 = ["a", "b", "x", "c"];
  const mr = L.llmCreate({ vocabSize: chars4.length, dModel: 4, heads: 2, dff: 6, maxLen: 8, rand: L.llmSeedRng(11) });
  const cr = L.llmConstrainedGenerate(mr, [2], chars4, L.llmRegexGrammar("ab"), 4, 1.0, 0,
    function () { return 0.5; }, 1);
  r.ok("采样约束生成只产出 \"ab\"",
    tEqBits(cr.tokens, ["a", "b"]) && cr.complete, JSON.stringify(cr.tokens));

  r.ok("文法入口的错误路径都报错",
    tThrowsMsg(function () { L.llmGrammarAllowed({ kind: "nope" }, [], jv); }, "无法识别") &&
    tThrowsMsg(function () { L.llmGrammarAccepts(null, []); }, "需要") &&
    tThrowsMsg(function () { L.llmGrammarMask(g, [], "abc"); }, "vocab") &&
    tThrowsMsg(function () { L.llmGrammarPick(g, [], [1], ["a", "b"]); }, "长度"));
}

/* KV 缓存增量前向 / 生成（Python _forward_step / _generate_cached 的 JS 版） */
function testIncrementalForward(L, r) {
  r.section("13. KV 缓存增量前向 / 生成");

  const mk = function (win) {
    return L.llmCreate({ vocabSize: 5, dModel: 4, heads: 2, dff: 6, maxLen: 6, rand: L.llmSeedRng(12345), window: win });
  };
  const m = mk(0);
  const ids = [1, 2, 3, 4];
  let c = L.llmPrefixCache(1, m.cfg.dModel);
  let diff = 0;
  for (let pos = 0; pos < ids.length; pos++) {
    const step = L.llmForwardStep(m, ids[pos], pos, c);
    const full = L.llmForward(m, ids.slice(0, pos + 1)).logits[pos];
    if (!tEqBits(step, full)) diff++;
  }
  r.ok("window=0 时单步前向与整段前向逐位一致", diff === 0, diff + " 处不同");

  const mw = mk(2);
  const wid = [1, 2, 3, 4, 0];
  let cw = L.llmPrefixCache(1, mw.cfg.dModel);
  let diffW = 0;
  for (let pos = 0; pos < wid.length; pos++) {
    const step = L.llmForwardStep(mw, wid[pos], pos, cw);
    const full = L.llmForward(mw, wid.slice(0, pos + 1)).logits[pos];
    if (!tEqBits(step, full)) diffW++;
  }
  r.ok("window=2 时单步前向与整段前向逐位一致", diffW === 0, diffW + " 处不同");

  /* 不滑动：cached 生成与整段重算逐 token 相同 */
  const a = L.llmGenerateCached(m, [1, 2], 4, 1.0, 0, function () { return 0.42; }, 1, {});
  const b = L.llmGenerate(m, [1, 2], 4, 1.0, 0, function () { return 0.42; }, 1, {});
  r.ok("cached 与整段生成逐 token 一致", tEqBits(a.ids, b), JSON.stringify([a.ids, b]));
  r.ok("prefill/steps 统计（无重建）",
    a.stats.prefill === 2 && a.stats.steps === 6 && a.stats.rebuilds === 0, JSON.stringify(a.stats));

  /* 滑动：maxLen=4、续写 6 个 token，窗口每步左移 → 重建 4 次 */
  const ms = L.llmCreate({ vocabSize: 5, dModel: 4, heads: 2, dff: 6, maxLen: 4, rand: L.llmSeedRng(4321), window: 0 });
  const a2 = L.llmGenerateCached(ms, [1, 2], 6, 1.0, 0, function () { return 0.42; }, 1, {});
  const b2 = L.llmGenerate(ms, [1, 2], 6, 1.0, 0, function () { return 0.42; }, 1, {});
  r.ok("窗口滑动后 cached 仍与整段生成逐 token 一致", tEqBits(a2.ids, b2),
    JSON.stringify([a2.ids, b2]));
  r.ok("滑动触发缓存重建", a2.stats.rebuilds === 4 && a2.stats.prefill === 2, JSON.stringify(a2.stats));

  r.ok("增量前向 / 生成的错误路径都报错",
    tThrowsMsg(function () { L.llmForwardStep(m, 9, 0, L.llmPrefixCache(1, m.cfg.dModel)); }, "非法") &&
    tThrowsMsg(function () { L.llmForwardStep(m, 1, 1, L.llmPrefixCache(1, m.cfg.dModel)); }, "不一致") &&
    tThrowsMsg(function () { L.llmForwardStep(m, 1, 0, L.llmPrefixCache(2, m.cfg.dModel)); }, "缓存") &&
    tThrowsMsg(function () { L.llmForwardStep(m, 1, 9, L.llmPrefixCache(1, m.cfg.dModel)); }, "pos") &&
    tThrowsMsg(function () { L.llmGenerateCached(m, [], 2, 1, 0, function () { return 0.5; }); }, "ids") &&
    tThrowsMsg(function () { L.llmGenerateCached(m, [1], -1, 1, 0, function () { return 0.5; }); }, "n"));
}
