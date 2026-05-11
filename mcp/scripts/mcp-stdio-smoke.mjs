/**
 * MCP stdio smoke: same framing as @modelcontextprotocol/sdk (newline-delimited JSON-RPC).
 * Fails if the server process exits before tools/list returns non-empty tools (catches import-time crashes).
 */
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SERVER_JS = path.join(__dirname, '..', 'dist', 'index.js');
const TIMEOUT_MS = 20_000;

function sendLine(stdin, obj) {
  stdin.write(`${JSON.stringify(obj)}\n`);
}

async function main() {
  if (!fs.existsSync(SERVER_JS)) {
    console.error(`smoke: missing ${SERVER_JS} — run npm run build first`);
    process.exit(1);
  }

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-mcp-smoke-'));
  const logosdbPath = path.join(tmp, '.logosdb');

  const child = spawn(process.execPath, [SERVER_JS], {
    cwd: tmp,
    env: {
      ...process.env,
      LOGOSDB_PATH: logosdbPath,
      LOGOSDB_INDEX_ROOT: tmp,
      NPM_CONFIG_FUND: 'false',
      NPM_CONFIG_AUDIT: 'false',
    },
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  let stderrBuf = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (c) => {
    stderrBuf += c;
  });

  let exitEarly = null;
  child.on('exit', (code, signal) => {
    if (code !== 0 && code !== null) {
      exitEarly = { code, signal };
    }
  });

  const rl = readline.createInterface({ input: child.stdout });

  const initId = 1;
  const listId = 2;

  sendLine(child.stdin, {
    jsonrpc: '2.0',
    id: initId,
    method: 'initialize',
    params: {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'logosdb-mcp-smoke', version: '0.0.0' },
    },
  });

  let phase = 'waitInit';
  let toolsOk = false;

  const fail = (msg) => {
    clearTimeout(timer);
    try {
      child.kill('SIGKILL');
    } catch {
      /* ignore */
    }
    console.error(msg);
    if (stderrBuf.trim()) {
      console.error('--- server stderr ---');
      console.error(stderrBuf.slice(-8000));
    }
    process.exit(1);
  };

  let settled = false;
  const timer = setTimeout(() => {
    if (!settled) {
      fail(`smoke: timeout ${TIMEOUT_MS}ms (no tools/list response)`);
    }
  }, TIMEOUT_MS);

  const readTask = (async () => {
    for await (const line of rl) {
      if (exitEarly) {
        fail(
          `smoke: server exited before success (code=${exitEarly.code} signal=${exitEarly.signal ?? 'none'})`,
        );
      }
      const trimmed = line.trim();
      if (!trimmed) continue;

      let msg;
      try {
        msg = JSON.parse(trimmed);
      } catch {
        continue;
      }

      if (msg.error) {
        fail(`smoke: JSON-RPC error: ${JSON.stringify(msg.error)}`);
      }

      if (phase === 'waitInit' && msg.id === initId && msg.result) {
        phase = 'sentList';
        sendLine(child.stdin, {
          jsonrpc: '2.0',
          method: 'notifications/initialized',
          params: {},
        });
        sendLine(child.stdin, {
          jsonrpc: '2.0',
          id: listId,
          method: 'tools/list',
          params: {},
        });
        continue;
      }

      if (msg.id === listId && msg.result?.tools && Array.isArray(msg.result.tools)) {
        const names = msg.result.tools.map((t) => t.name);
        const need = ['logosdb_search', 'logosdb_index_file', 'logosdb_list'];
        const missing = need.filter((n) => !names.includes(n));
        if (missing.length) {
          fail(`smoke: tools/list missing tools: ${missing.join(', ')} (got ${names.length} tools)`);
        }
        toolsOk = true;
        return;
      }
    }

    if (!toolsOk) {
      fail('smoke: stdout closed before tools/list succeeded');
    }
  })();

  try {
    await readTask;
  } catch (e) {
    clearTimeout(timer);
    fail(`smoke: ${e instanceof Error ? e.message : String(e)}`);
  }
  settled = true;
  clearTimeout(timer);

  child.stdin.end();
  try {
    child.kill('SIGTERM');
  } catch {
    /* ignore */
  }

  fs.rmSync(tmp, { recursive: true, force: true });

  if (!toolsOk) {
    fail('smoke: internal state toolsOk=false');
  }

  console.log(`smoke: ok (${TIMEOUT_MS}ms budget) — tools/list returned logosdb_search, logosdb_index_file, logosdb_list`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
