#!/usr/bin/env node
/**
 * Structural check for the Codex plugin package.
 * Confirms the portable manifest, the stdio MCP entry, the skill, and that
 * the wrapper pins the same logosdb-mcp-server version as mcp/package.json.
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const pluginRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.resolve(pluginRoot, '../..');
const errors = [];

function fail(msg) {
  errors.push(msg);
}

function readJson(rel) {
  const p = path.join(pluginRoot, rel);
  if (!fs.existsSync(p)) {
    fail(`missing ${rel}`);
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch (err) {
    fail(`${rel} is not JSON: ${err.message}`);
    return null;
  }
}

function assertInside(rel, value) {
  if (typeof value !== 'string' || !value.startsWith('./') || value.includes('..')) {
    fail(`${rel} must be a ./ path with no .. (got ${JSON.stringify(value)})`);
  }
}

const plugin = readJson('plugin.json');
if (plugin) {
  if (plugin.$schema !== 'https://agent-plugins.org/schemas/1.0.0/plugin.schema.json') {
    fail('plugin.json $schema is not the Agent Plugins 1.0.0 schema');
  }
  if (plugin.name !== 'codex-semantic-memory') fail('plugin.json name');
  if (!/^\d+\.\d+\.\d+$/.test(plugin.version ?? '')) fail('plugin.json version');
  if (!plugin.author || typeof plugin.author.name !== 'string' || !plugin.author.name) {
    fail('plugin.json author.name');
  }
  const iface = plugin.extensions?.['com.openai']?.interface;
  if (!iface?.displayName || !iface.shortDescription || !iface.longDescription) {
    fail('plugin.json extensions.com.openai.interface');
  }
  if (iface?.displayName && iface.displayName.length > 30) fail('displayName longer than 30');
  if (iface?.shortDescription && iface.shortDescription.length > 30) {
    fail('shortDescription longer than 30');
  }
  for (const key of ['composerIcon', 'logo']) {
    if (!iface?.[key]) fail(`interface.${key} missing`);
    else {
      assertInside(`interface.${key}`, iface[key]);
      const icon = path.join(pluginRoot, iface[key].slice(2));
      if (!fs.existsSync(icon)) fail(`missing icon ${iface[key]}`);
    }
  }
  if (plugin.extensions?.['com.openai']?.requires_local_executor !== true) {
    fail('requires_local_executor must be true');
  }
}

const mcp = readJson('mcp.json');
const server = mcp?.mcpServers?.logosdb;
if (!server || server.type !== 'stdio' || server.command !== 'bash') {
  fail('mcp.json logosdb must be a stdio bash server');
}
const arg = server?.args?.[0] ?? '';
if (!arg.endsWith('/scripts/logosdb-mcp-wrap.sh')) {
  fail('mcp.json args must launch scripts/logosdb-mcp-wrap.sh');
}

const compat = readJson('.mcp.json');
if (compat?.mcpServers?.logosdb?.command !== 'bash') fail('.mcp.json command');
if (compat?.mcpServers?.logosdb?.type !== 'stdio') fail('.mcp.json type');

const overlay = readJson('.codex-plugin/plugin.json');
if (overlay) {
  assertInside('skills', overlay.skills);
  assertInside('mcpServers', overlay.mcpServers);
  if (overlay.skills !== './skills/') fail('Codex overlay skills must be ./skills/');
  if (overlay.mcpServers !== './.mcp.json') fail('Codex overlay mcpServers must be ./.mcp.json');
  if (overlay.version !== plugin?.version) fail('overlay version differs from plugin.json');
}

const marketPath = path.join(repoRoot, '.agents/plugins/marketplace.json');
if (!fs.existsSync(marketPath)) fail('missing .agents/plugins/marketplace.json');
else {
  const market = JSON.parse(fs.readFileSync(marketPath, 'utf8'));
  if (market.name !== 'logosdb') fail('marketplace name');
  const entry = (market.plugins ?? []).find((p) => p.name === 'codex-semantic-memory');
  if (!entry) fail('marketplace missing codex-semantic-memory');
  else if (entry.source?.path !== './plugins/codex-semantic-memory') {
    fail('marketplace source.path');
  }
}

const skillPath = path.join(pluginRoot, 'skills/semantic-memory/SKILL.md');
const skill = fs.readFileSync(skillPath, 'utf8');
const fm = skill.match(/^---\n([\s\S]*?)\n---/);
if (!fm) fail('SKILL.md front matter');
else {
  const name = fm[1].match(/^name:\s*(\S+)\s*$/m);
  const desc = fm[1].match(/^description:\s*>?\s*([\s\S]*?)(?=\n[a-z]|$(?![\s\S]))/m);
  if (!name || name[1] !== 'semantic-memory') fail('skill name');
  const description = (desc?.[1] ?? '').replace(/\s+/g, ' ').trim();
  if (!description) fail('skill description');
  if (description.length > 1024) fail('skill description longer than 1024');
}
if (!skill.includes('chunking') || !skill.includes('"auto"')) {
  fail('SKILL.md must call logosdb_index_file with chunking auto');
}

const agents = fs.readFileSync(path.join(pluginRoot, 'AGENTS.md'), 'utf8');
if (!agents.includes('chunking = "auto"')) fail('AGENTS.md must request chunking auto');

const wrapPath = path.join(pluginRoot, 'scripts/logosdb-mcp-wrap.sh');
const wrap = fs.readFileSync(wrapPath, 'utf8');
const pin = wrap.match(/PIN="(\d+\.\d+\.\d+)"/);
const pkg = JSON.parse(fs.readFileSync(path.join(repoRoot, 'mcp/package.json'), 'utf8'));
if (!pin || pin[1] !== pkg.version) {
  fail(`wrapper pin ${pin?.[1]} != mcp/package.json ${pkg.version}`);
}
if (!wrap.includes('LOGOSDB_INDEX_ROOT')) fail('wrapper must set LOGOSDB_INDEX_ROOT');
try {
  execFileSync('bash', ['-n', wrapPath], { stdio: 'pipe' });
} catch (err) {
  fail(`wrap.sh syntax: ${err.message}`);
}

if (fs.existsSync(path.join(pluginRoot, '.codex-plugin/marketplace.json'))) {
  fail('remove .codex-plugin/marketplace.json; the catalog is .agents/plugins/marketplace.json');
}

if (errors.length) {
  console.error(errors.map((e) => `- ${e}`).join('\n'));
  process.exit(1);
}
console.log(`codex plugin ok (${plugin.version}, logosdb-mcp-server ${pkg.version})`);
