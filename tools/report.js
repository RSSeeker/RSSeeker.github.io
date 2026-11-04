"use strict";
/* 极简检查报告器：逐条 PASS/FAIL，最后按失败数决定退出码。 */
function createReporter() {
  let checks = 0;
  let failures = 0;

  function ok(name, cond, extra) {
    checks += 1;
    if (!cond) failures += 1;
    console.log((cond ? "PASS " : "FAIL ") + name + (extra ? "  " + extra : ""));
  }

  function section(title) {
    console.log("\n== " + title + " ==");
  }

  function note(text) {
    console.log("   " + text);
  }

  return {
    ok,
    section,
    note,
    get checks() { return checks; },
    get failures() { return failures; },
  };
}

/* 唯一的汇总出口：成功时正好一行 "ALL PASS (N checks)" */
function summarize(checks, failures) {
  if (failures) {
    console.log("\n" + failures + " FAILURE(S) / " + checks + " checks");
    return 1;
  }
  console.log("\nALL PASS (" + checks + " checks)");
  return 0;
}

module.exports = { createReporter, summarize };
