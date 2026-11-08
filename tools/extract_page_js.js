"use strict";
/*
 * 公共样板：从 index.html 抽取内联 JS，并在 Node 的伪 DOM 里执行。
 * tools/llm_core_test.js 与 tools/llm_page_check.js 共用，避免各自复制粘贴。
 *
 * 所有路径都以本文件为基准解析（仓库根 = tools/..），不依赖当前工作目录，
 * 也不写任何绝对路径。
 */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const INDEX_PATH = path.join(ROOT, "index.html");

/* 页内 LLM 核心对外导出的函数；缺失任何一个都说明抽取边界或页面结构变了 */
const LLM_CORE_FUNCTIONS = [
  "llmSeedRng", "llmGauss", "llmZeros", "llmZerosLike", "llmGaussMat", "llmPosTable",
  "llmCreate", "mv", "mvT", "addVec", "addInto", "dot", "softmaxRow", "lnForward", "lnBackward",
  "llmForward", "llmBackward", "llmZeroGrad", "llmAdamStep", "llmTrainStep",
  "llmGenOptions", "llmApplyPenalties", "llmMaskMinP",
  "llmKeepMask", "llmArgmax", "llmMaskTypical", "llmMaskTopA", "llmMaskEta",
  "llmMaskTailFree", "llmMirostatFilter", "llmMirostatUpdate",
  "llmApplyFilters", "llmDraw",
  "llmPrefixCache", "llmCacheCheckRows", "llmCacheCheckLayer", "llmCacheSet",
  "llmCacheGet", "llmCacheLength", "llmCacheTruncate", "llmCacheMerge",
  "llmCacheToJSON", "llmCacheFromJSON", "llmCacheStats", "llmCacheBars",
  "llmJsonStart", "llmJsonTokenKind", "llmJsonStep", "llmJsonAllowed",
  "llmJsonMask", "llmJsonPick", "llmJsonComplete",
  "llmSampleToken", "llmGenerate", "llmPerplexity", "llmExportModel", "llmImportModel",
  "llmCheckVec", "llmCos", "llmContrastiveSelect", "llmContrastiveSearch",
  "llmRegexCharPredicate", "llmRegexNewNfa", "llmRegexNewState", "llmRegexAdd",
  "llmRegexAddEps", "llmRegexClosure", "llmRegexStep", "llmRegexLive",
  "llmRegexParse", "llmRegexCompile", "llmRegexGrammar", "llmRegexAdvance",
  "llmRegexState", "llmRegexViable", "llmRegexAccepts", "llmRegexAllowed", "llmRegexChoices",
  "llmCfgGrammar", "llmCfgMatch", "llmCfgChart", "llmCfgAccepts", "llmCfgExpected", "llmCfgAllowed",
  "llmJsonCharGrammar",
  "llmGrammarAllowed", "llmGrammarAccepts", "llmGrammarMask", "llmGrammarPick",
  "llmGrammarGenerate", "llmConstrainedGenerate",
  "llmForwardStep", "llmGenerateCached",
];

function readIndexHtml() {
  return fs.readFileSync(INDEX_PATH, "utf8");
}

function lineOf(html, index) {
  return html.slice(0, index).split("\n").length;
}

/* 所有不带 src 的内联 <script>；line 是起始行号（1 起） */
function inlineScripts(html) {
  const re = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
  const out = [];
  let m;
  while ((m = re.exec(html)) !== null) {
    if (/\bsrc\s*=/.test(m[1] || "")) continue;
    out.push({ code: m[2], line: lineOf(html, m.index) });
  }
  return out;
}

/* 从 "function llmSeedRng" 到面板逻辑标记之间的全部核心代码 */
function extractLlmCore(html) {
  const start = html.indexOf("function llmSeedRng");
  if (start < 0) throw new Error("index.html 里找不到 function llmSeedRng");
  const end = html.indexOf("/* ---------------- 面板逻辑", start);
  if (end < 0) throw new Error("index.html 里找不到「面板逻辑」标记");
  return {
    code: html.slice(start, end),
    start,
    end,
    startLine: lineOf(html, start),
    endLine: lineOf(html, end),
  };
}

/* 页面底部「因果语言模型」整段 IIFE */
function extractLlmIife(html) {
  const banner = "/* ===================== 因果语言模型（字符级 GPT）";
  const at = html.indexOf(banner);
  if (at < 0) throw new Error("index.html 里找不到「因果语言模型」段落");
  const scriptEnd = html.indexOf("</script>", at);
  if (scriptEnd < 0) throw new Error("「因果语言模型」段落缺少 </script>");
  const start = html.indexOf("(function () {", at);
  if (start < 0 || start > scriptEnd) throw new Error("「因果语言模型」段落里找不到 IIFE 开头");
  const close = html.lastIndexOf("})();", scriptEnd);
  if (close < start) throw new Error("「因果语言模型」段落里找不到 IIFE 结尾");
  const end = close + 5;
  return {
    code: html.slice(start, end),
    start,
    end,
    startLine: lineOf(html, start),
    endLine: lineOf(html, end),
  };
}

/* 把核心代码当 CommonJS 模块执行，返回 LLM_CORE_FUNCTIONS 里全部函数组成的对象。
   用 new Function 而不是 vm 新上下文：异常仍是本进程的 Error，检查里可以 instanceof。 */
function loadLlmCore(html) {
  const src = html === undefined ? readIndexHtml() : html;
  const { code, startLine } = extractLlmCore(src);
  const missing = LLM_CORE_FUNCTIONS.filter(n => !new RegExp("function\\s+" + n + "\\b").test(code));
  if (missing.length) {
    throw new Error("第 " + startLine + " 行起的 LLM 核心缺少函数：" + missing.join(", "));
  }
  const exportCode = "\nmodule.exports = {" +
    LLM_CORE_FUNCTIONS.map(n => n + ": " + n).join(", ") + "};\n";
  const factory = new Function("module", code + exportCode);
  const mod = { exports: {} };
  factory(mod);
  return mod.exports;
}

/* 极简伪 DOM：够页面里的 $id(...) / classList / 事件 / 文件读取 / Blob 使用。
   hidden 里的 id 会让 getElementById 返回 null，用来模拟旧页面缺元素。 */
function createFakeDom() {
  const els = new Map();
  const hidden = new Set();
  const captured = { blobText: null, blobType: null };

  function makeEl(id) {
    const listeners = {};
    return {
      id, value: "", textContent: "", disabled: false, title: "",
      style: {}, files: null, parentNode: null, clicked: false,
      classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
      addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
      trigger(type) { (listeners[type] || []).forEach(fn => fn({ target: this })); },
      click() { this.clicked = true; },
      appendChild(child) { child.parentNode = this; },
      removeChild(child) { child.parentNode = null; },
    };
  }

  const document = {
    getElementById(id) {
      if (hidden.has(id)) return null;
      if (!els.has(id)) els.set(id, makeEl(id));
      return els.get(id);
    },
    createElement() { return makeEl("a"); },
    body: makeEl("body"),
  };

  return {
    document,
    els,
    hidden,
    captured,
    get(id) { return document.getElementById(id); },
  };
}

/* 装浏览器全局（document / Blob / FileReader / URL），返回 restore() */
function installBrowserGlobals(fake) {
  const saved = {
    document: global.document,
    Blob: global.Blob,
    FileReader: global.FileReader,
    URL: global.URL,
  };

  class FakeBlob {
    constructor(parts, opts) {
      fake.captured.blobText = (parts || []).join("");
      fake.captured.blobType = opts && opts.type;
    }
  }
  class FakeFileReader {
    readAsText(file) {
      this.result = file.text;
      if (this.onload) setTimeout(() => this.onload(), 0);
    }
  }

  global.document = fake.document;
  global.Blob = FakeBlob;
  global.FileReader = FakeFileReader;
  global.URL = {
    createObjectURL() { return "blob:fake"; },
    revokeObjectURL() {},
  };

  return function restore() {
    if (saved.document === undefined) delete global.document; else global.document = saved.document;
    if (saved.Blob === undefined) delete global.Blob; else global.Blob = saved.Blob;
    if (saved.FileReader === undefined) delete global.FileReader; else global.FileReader = saved.FileReader;
    if (saved.URL === undefined) delete global.URL; else global.URL = saved.URL;
  };
}

/* 在伪 DOM 里执行页面里的 LLM 面板 IIFE；失败时也会恢复全局 */
function loadLlmPanel(html) {
  const src = html === undefined ? readIndexHtml() : html;
  const iife = extractLlmIife(src);
  const fake = createFakeDom();
  const restore = installBrowserGlobals(fake);
  try {
    new Function(iife.code)();
  } catch (e) {
    restore();
    throw e;
  }
  fake.iife = iife;
  fake.restore = restore;
  return fake;
}

module.exports = {
  ROOT,
  INDEX_PATH,
  LLM_CORE_FUNCTIONS,
  readIndexHtml,
  lineOf,
  inlineScripts,
  extractLlmCore,
  extractLlmIife,
  loadLlmCore,
  createFakeDom,
  installBrowserGlobals,
  loadLlmPanel,
};
