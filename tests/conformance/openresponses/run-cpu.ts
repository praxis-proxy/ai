// Thin execution adapter: retain upstream templates and validators unchanged,
// but give slow CPU inference an explicit request timeout.
import { pathToFileURL } from "node:url";

const [suite, baseUrl, model, timeout, ids] = Bun.argv.slice(2);
const timeoutMs = Number(timeout);
if (!suite || !baseUrl || !model || !ids || !Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) {
  throw new Error("expected suite directory, base URL, model, positive timeout milliseconds, and template IDs");
}

const { testTemplates, runAllTests } = await import(
  pathToFileURL(`${suite}/src/lib/compliance-tests.ts`).href
);
const requested = new Set(ids.split(","));
const selected = testTemplates.filter((template: { id: string }) => requested.has(template.id));
if (selected.length !== requested.size) {
  throw new Error("unknown OpenResponses template ID");
}
const results = await runAllTests({
  baseUrl, model, timeoutMs,
  apiKey: "test",
  authHeaderName: "Authorization",
  useBearerPrefix: true,
  runtime: "server",
}, () => {}, selected);
const count = (status: string) => results.filter((result: { status: string }) => result.status === status).length;
const summary = { passed: count("passed"), failed: count("failed"), skipped: count("skipped"), total: results.length };
console.log(JSON.stringify({ summary, results }, null, 2));
process.exit(summary.passed === selected.length ? 0 : 1);
