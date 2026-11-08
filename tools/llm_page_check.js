"use strict";
/*
 * 页面级检查：index.html 的结构（脚本语法 / 标签配平 / id 契约），
 * 以及在伪 DOM 里执行页面真正要发布的 LLM 面板代码。
 *
 * 与 tools/llm_core_test.js 共用 tools/extract_page_js.js 的抽取与伪 DOM 样板。
 */
const vm = require("vm");
const extract = require("./extract_page_js.js");
const { createReporter, summarize } = require("./report.js");

function waitFor(fn, ms) {
  return new Promise((resolve) => {
    const t0 = Date.now();
    (function poll() {
      let v = false;
      try { v = fn(); } catch (e) { v = false; }
      if (v) return resolve(true);
      if (Date.now() - t0 > (ms || 5000)) return resolve(false);
      setTimeout(poll, 10);
    })();
  });
}

/* 逐个切出开始标签（跳过注释 / script / style / doctype），供配平与 id 检查用 */
function tagTokens(src) {
  const out = [];
  let i = 0;
  while (i < src.length) {
    const lt = src.indexOf("<", i);
    if (lt < 0) break;
    if (src.startsWith("<!--", lt)) {
      const e = src.indexOf("-->", lt + 4);
      i = e < 0 ? src.length : e + 3;
      continue;
    }
    if (/^<script\b/i.test(src.slice(lt, lt + 8))) {
      const e = src.toLowerCase().indexOf("</script", lt);
      i = e < 0 ? src.length : src.indexOf(">", e) + 1;
      continue;
    }
    if (/^<style\b/i.test(src.slice(lt, lt + 7))) {
      const e = src.toLowerCase().indexOf("</style", lt);
      i = e < 0 ? src.length : src.indexOf(">", e) + 1;
      continue;
    }
    if (src[lt + 1] === "!" || src[lt + 1] === "?") { i = src.indexOf(">", lt) + 1; continue; }
    let j = lt + 1, quote = null;
    while (j < src.length) {
      const c = src[j];
      if (quote) { if (c === quote) quote = null; }
      else if (c === '"' || c === "'") quote = c;
      else if (c === ">") break;
      j++;
    }
    out.push({ raw: src.slice(lt, j + 1), at: lt });
    i = j + 1;
  }
  return out;
}

async function run() {
  const r = createReporter();
  const html = extract.readIndexHtml();

  r.section("1. 文件基本属性");
  r.ok("index.html 非空", html.length > 100000, html.length + " 字符");
  /* 不要求必须是 LF：Windows 上 core.autocrlf=true 的克隆会得到 CRLF，
     页面在两种行尾下行为一致。只拒绝混用，避免 diff/行号工具出错。 */
  const crlfCount = (html.match(/\r\n/g) || []).length;
  const lfOnlyCount = (html.match(/\n/g) || []).length - crlfCount;
  r.ok("行尾风格一致（无 LF/CRLF 混用）", crlfCount === 0 || lfOnlyCount === 0,
    crlfCount + " 个 CRLF / " + lfOnlyCount + " 个单 LF");

  /* ---------- 2. 每个内联 <script> 单独过语法编译 ---------- */
  r.section("2. 内联脚本语法");
  const scripts = extract.inlineScripts(html);
  r.ok("找到内联脚本", scripts.length >= 1, scripts.length + " 个");
  scripts.forEach((s, i) => {
    let err = null;
    try {
      new vm.Script(s.code, { filename: "index_inline_" + i + ".js" });
    } catch (e) {
      err = e;
    }
    r.ok("内联脚本 #" + i + "（起始行 " + s.line + "）语法通过", !err,
      err ? String(err.message).split("\n")[0] : "");
  });

  /* ---------- 3. 标签配对 ---------- */
  r.section("3. 标签配平");
  const tokens = tagTokens(html);
  const WATCH = ["div", "span", "button", "textarea", "table", "select", "option"];
  const stack = [];
  const counts = {};
  let balanceError = null;
  for (const t of tokens) {
    const mm = /^<\s*(\/?)\s*([a-zA-Z][\w:-]*)/.exec(t.raw);
    if (!mm) continue;
    const closing = mm[1] === "/";
    const name = mm[2].toLowerCase();
    if (WATCH.indexOf(name) < 0) continue;
    const selfClosing = /\/\s*>$/.test(t.raw);
    const line = extract.lineOf(html, t.at);
    if (!counts[name]) counts[name] = { open: 0, close: 0 };
    if (closing) {
      counts[name].close++;
      const top = stack.pop();
      if (!top || top.name !== name) {
        balanceError = "第 " + line + " 行 </" + name + "> 与 " +
          (top ? "第 " + top.line + " 行 <" + top.name + ">" : "空栈") + " 不匹配";
        break;
      }
    } else if (!selfClosing) {
      counts[name].open++;
      stack.push({ name, line });
    }
  }
  if (balanceError) r.note(balanceError);
  else if (stack.length) r.note("未闭合：" + stack.map(s => "<" + s.name + ">@" + s.line).join(", "));
  r.ok("div/span/button/textarea/table/select/option 全部配平",
    !balanceError && stack.length === 0,
    WATCH.map(n => n + " " + (counts[n] ? counts[n].open + "/" + counts[n].close : "0/0")).join(" "));

  /* ---------- 4. 元素 id 与 $id 引用 ---------- */
  r.section("4. 元素 id 契约");
  const ids = [];
  for (const t of tokens) {
    if (/^<\s*\//.test(t.raw)) continue;
    const idm = /\bid\s*=\s*("([^"]*)"|'([^']*)')/.exec(t.raw);
    if (idm) {
      ids.push({
        id: idm[2] !== undefined ? idm[2] : idm[3],
        line: extract.lineOf(html, t.at),
      });
    }
  }
  const seen = new Map(), dup = [];
  for (const { id, line } of ids) {
    if (seen.has(id)) dup.push(id + "（第 " + seen.get(id) + "、" + line + " 行）");
    else seen.set(id, line);
  }
  r.ok("所有 id 唯一", dup.length === 0,
    "共 " + ids.length + " 个 id" + (dup.length ? "，重复：" + dup.join(", ") : ""));

  const refs = new Set();
  let rm;
  const refRe = /(?:\$id|document\.getElementById)\(\s*"([^"]+)"\s*\)/g;
  while ((rm = refRe.exec(html)) !== null) refs.add(rm[1]);
  const missing = [...refs].filter(id => !seen.has(id));
  r.ok("所有 $id(...) 引用的元素都存在", missing.length === 0, missing.join(", ") || "");

  /* ---------- 5. 伪 DOM 执行 LLM 面板 IIFE ---------- */
  r.section("5. 伪 DOM：训练 / 生成 / 保存 / 加载");
  let panel;
  try {
    panel = extract.loadLlmPanel(html);
  } catch (e) {
    r.ok("IIFE 在伪 DOM 中执行不抛错", false, e && e.message);
    return { checks: r.checks, failures: r.failures };
  }

  try {
    const iife = panel.iife;
    r.ok("截取的 IIFE 以 (function 开头、})(); 结尾",
      iife.code.startsWith("(function () {") && iife.code.trim().endsWith("})();"),
      "第 " + iife.startLine + "–" + iife.endLine + " 行, " + iife.code.length + " 字符");
    r.ok("IIFE 含新核心函数",
      ["llmSampleToken", "llmPerplexity", "llmExportModel", "llmImportModel", "llmReadWindow"]
        .every(n => iife.code.indexOf(n) >= 0));
    r.ok("IIFE 在伪 DOM 中执行不抛错", true);

    const el = id => panel.get(id);
    el("llmCorpus").value = "abcabcabcabc";
    el("llmEpochs").value = "6";
    el("llmLr").value = "0.05";
    el("llmDim").value = "8";
    el("llmWindow").value = "0";
    el("llmPrompt").value = "ab";
    el("llmNew").value = "8";
    el("llmTemp").value = "0.8";
    el("llmTopP").value = "1";
    el("llmStats").textContent = "";

    el("llmTrainBtn").trigger("click");
    const trained = await waitFor(() =>
      el("llmProgressText").textContent.indexOf("训练完成") >= 0 &&
      el("llmOut").textContent.length > 0, 8000);
    r.ok("点「训练」后训练完成并自动生成", trained, el("llmProgressText").textContent);

    const stats1 = el("llmStats").textContent;
    r.ok("统计行含 困惑度", /困惑度 (\d+\.\d+|inf)/.test(stats1), stats1);
    r.ok("统计行含 窗口 全", stats1.indexOf("窗口 全") >= 0, stats1);
    r.ok("统计行含 参数", /参数 \d+/.test(stats1), stats1);

    const out1 = el("llmOut").textContent;
    r.ok("生成文本长度为 prompt 2 + new 8", out1.length === 10, JSON.stringify(out1));
    r.ok("生成文本只含词表字符", Array.from(out1).every(c => "abc".indexOf(c) >= 0));

    /* 保存 */
    el("llmTopP").value = "0.5";
    el("llmGenBtn").trigger("click");
    r.ok("top_p=0.5 下生成仍可用", el("llmOut").textContent.length === 10,
      JSON.stringify(el("llmOut").textContent));
    /* 采样控制：留空按中性默认处理；固定时间种子后与显式中性逐位一致 */
    const realNow = Date.now;
    const GEN_IDS = ["llmMinP", "llmRepPenalty", "llmPresence", "llmFrequency", "llmNgram"];
    const NEUTRAL = ["0", "1", "0", "0", "0"];
    Date.now = () => 1700000000000;
    GEN_IDS.forEach(id => { el(id).value = ""; });
    el("llmTopP").value = "1";
    el("llmGenBtn").trigger("click");
    const outBlank = el("llmOut").textContent;
    GEN_IDS.forEach((id, i) => { el(id).value = NEUTRAL[i]; });
    el("llmGenBtn").trigger("click");
    const outNeutral = el("llmOut").textContent;
    r.ok("五项控制留空与显式中性参数生成结果逐位一致",
      outBlank.length === 10 && outBlank === outNeutral,
      JSON.stringify(outBlank) + " vs " + JSON.stringify(outNeutral));

    /* 面板接线：惩罚 + ngram=2 仍然能生成，且不出现重复二元组 */
    el("llmNew").value = "4";
    el("llmMinP").value = "0.9";
    el("llmRepPenalty").value = "2";
    el("llmPresence").value = "1";
    el("llmFrequency").value = "1";
    el("llmNgram").value = "2";
    el("llmGenBtn").trigger("click");
    const outPen = el("llmOut").textContent;
    Date.now = realNow;
    const bigrams = new Set(["a,b"]);
    let panelNoRepeat = true;
    for (let i = 1; i < outPen.length; i++) {
      const k = outPen[i - 1] + outPen[i];
      if (bigrams.has(k)) panelNoRepeat = false;
      bigrams.add(k);
    }
    r.ok("面板惩罚 + ngram=2 下仍生成且无重复二元组",
      outPen.length === 6 && panelNoRepeat, JSON.stringify(outPen));

    /* ---------- 三项新能力接线：KV 缓存 / 对比搜索 / 语法约束 ---------- */
    el("llmMinP").value = "0";
    el("llmRepPenalty").value = "1";
    el("llmPresence").value = "0";
    el("llmFrequency").value = "0";
    el("llmNgram").value = "0";
    el("llmTemp").value = "1";
    el("llmTopP").value = "1";
    el("llmNew").value = "6";

    el("llmUseCache").checked = true;
    el("llmGenBtn").trigger("click");
    r.ok("勾选 KV 缓存后生成正常", el("llmOut").textContent.length === 8,
      JSON.stringify(el("llmOut").textContent));
    r.ok("缓存统计写入统计行", el("llmStats").textContent.indexOf("KV 缓存 prefill") >= 0,
      el("llmStats").textContent);
    el("llmUseCache").checked = false;

    el("llmTopK").value = "2";
    el("llmAlpha").value = "0.5";
    el("llmContrastiveBtn").trigger("click");
    r.ok("对比搜索生成正常", el("llmOut").textContent.length === 8,
      JSON.stringify(el("llmOut").textContent));
    r.ok("对比搜索统计行", el("llmStats").textContent.indexOf("对比搜索") >= 0,
      el("llmStats").textContent);
    el("llmTopK").value = "0";
    el("llmContrastiveBtn").trigger("click");
    r.ok("对比 top_k=0 面板报错",
      el("llmProgressText").textContent.indexOf("top_k") >= 0, el("llmProgressText").textContent);
    el("llmTopK").value = "2";
    el("llmAlpha").value = "1.5";
    el("llmContrastiveBtn").trigger("click");
    r.ok("对比 alpha=1.5 面板报错",
      el("llmProgressText").textContent.indexOf("alpha") >= 0, el("llmProgressText").textContent);
    el("llmAlpha").value = "0.6";

    el("llmConstraint").value = "a(b|c)*";
    el("llmConstraintBtn").trigger("click");
    const outConstrained = el("llmOut").textContent;
    r.ok("约束解码生成正常", outConstrained.length > 0 &&
      Array.from(outConstrained).every(c => "abc".indexOf(c) >= 0), JSON.stringify(outConstrained));
    r.ok("约束解码统计行", el("llmStats").textContent.indexOf("约束解码") >= 0,
      el("llmStats").textContent);
    el("llmConstraint").value = "a(";
    el("llmConstraintBtn").trigger("click");
    r.ok("非法约束文法面板报错",
      el("llmProgressText").textContent.indexOf("约束文法非法") >= 0, el("llmProgressText").textContent);
    el("llmConstraint").value = "json";

    /* 非法采样参数：面板内报错且不生成 */
    el("llmMinP").value = "1.5";
    el("llmGenBtn").trigger("click");
    r.ok("面板 min_p=1.5 报错", el("llmProgressText").textContent.indexOf("min_p") >= 0,
      el("llmProgressText").textContent);
    el("llmMinP").value = "0";

    el("llmNgram").value = "1";
    el("llmGenBtn").trigger("click");
    r.ok("面板 no_repeat_ngram_size=1 报错",
      el("llmProgressText").textContent.indexOf("no_repeat_ngram_size") >= 0,
      el("llmProgressText").textContent);
    el("llmNgram").value = "0";

    el("llmPresence").value = "abc";
    el("llmGenBtn").trigger("click");
    r.ok("面板非数值 penalty 报错",
      el("llmProgressText").textContent.indexOf("presence_penalty 必须是数字") >= 0,
      el("llmProgressText").textContent);

    /* 复原为中性默认值，避免影响后续「缺 llmWindow」用例 */
    GEN_IDS.forEach(id => { el(id).value = ""; });
    el("llmNew").value = "8";

    el("llmSaveBtn").trigger("click");
    r.ok("保存写出了 JSON Blob", !!panel.captured.blobText,
      (panel.captured.blobText || "").length + " 字节");
    let saved = null;
    try { saved = JSON.parse(panel.captured.blobText); } catch (e) {}
    r.ok("保存的 JSON 可解析且 format 正确", !!saved && saved.format === "llm_web_v1");
    r.ok("保存含 config/vocab/params",
      !!saved && saved.config.window === 0 && saved.vocab.length === 3 && saved.params.E.length === 3,
      saved ? ("vocab=" + saved.vocab.join("") + " E=" + saved.params.E.length) : "");

    /* 非法窗口 */
    el("llmWindow").value = "1.5";
    el("llmProgressText").textContent = "";
    el("llmTrainBtn").trigger("click");
    r.ok("窗口输入 1.5 报错且不训练",
      el("llmProgressText").textContent.indexOf("窗口必须是非负整数") >= 0,
      el("llmProgressText").textContent);
    el("llmWindow").value = "0";

    /* 非法 JSON 加载 */
    el("llmLoadFile").files = [{ text: "{oops" }];
    el("llmLoadFile").trigger("change");
    await waitFor(() => el("llmProgressText").textContent.indexOf("加载失败") >= 0, 2000);
    r.ok("非法 JSON 给出「加载失败：文件不是合法 JSON」",
      el("llmProgressText").textContent.indexOf("加载失败：文件不是合法 JSON") >= 0,
      el("llmProgressText").textContent);

    /* 正常加载 */
    el("llmLoadFile").files = [{ text: panel.captured.blobText }];
    el("llmLoadFile").trigger("change");
    const loaded = await waitFor(() => el("llmStats").textContent.indexOf("已加载模型") === 0, 2000);
    r.ok("合法存档加载成功", loaded, el("llmStats").textContent);
    r.ok("加载后统计行含 词表 3 / 窗口 全",
      /词表 3 /.test(el("llmStats").textContent) && el("llmStats").textContent.indexOf("窗口 全") >= 0,
      el("llmStats").textContent);
    el("llmGenBtn").trigger("click");
    r.ok("加载后可继续生成", el("llmOut").textContent.length === 10,
      JSON.stringify(el("llmOut").textContent));

    /* 旧页面没有 #llmWindow 时的防御默认值 */
    panel.hidden.add("llmWindow");
    el("llmStats").textContent = "";
    el("llmTrainBtn").trigger("click");
    const trained2 = await waitFor(() =>
      el("llmStats").textContent.indexOf("窗口 全") >= 0, 8000);
    r.ok("缺少 #llmWindow 时按全注意力训练（防御默认）", trained2, el("llmStats").textContent);
    panel.hidden.delete("llmWindow");
  } finally {
    panel.restore();
  }

  return { checks: r.checks, failures: r.failures };
}

module.exports = { run };

if (require.main === module) {
  run().then(
    res => { process.exitCode = summarize(res.checks, res.failures); },
    err => { console.log("FAIL 套件异常: " + (err && err.stack ? err.stack : err)); process.exitCode = 1; }
  );
}
