#!/usr/bin/env node
"use strict";
/*
 * 本地校验入口：node tools/check.js
 *
 * 依次跑两个套件（都在本进程内完成），退出码 0 = 全部通过。
 * 这些脚本只在开发时使用，不参与页面运行。
 */
const { summarize } = require("./report.js");

const MIN_NODE_MAJOR = 18;
const SUITES = [
  ["LLM 核心数值（核心代码从 index.html 抽取）", require("./llm_core_test.js")],
  ["页面结构 + 伪 DOM 集成", require("./llm_page_check.js")],
];

async function main() {
  const major = Number(process.versions.node.split(".")[0]);
  if (!Number.isFinite(major) || major < MIN_NODE_MAJOR) {
    console.error("需要 Node " + MIN_NODE_MAJOR + "+，当前版本 " + process.versions.node);
    return 1;
  }

  let checks = 0;
  let failures = 0;
  for (const [name, suite] of SUITES) {
    console.log("\n######## " + name + " ########");
    try {
      const res = await suite.run();
      checks += res.checks;
      failures += res.failures;
    } catch (err) {
      failures += 1;
      console.log("FAIL 套件异常: " + (err && err.stack ? err.stack : err));
    }
  }
  return summarize(checks, failures);
}

main().then(
  code => { process.exitCode = code; },
  err => {
    console.log("FAIL 入口异常: " + (err && err.stack ? err.stack : err));
    process.exitCode = 1;
  }
);
