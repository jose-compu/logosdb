import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { applyCli, consumeDbPath, formatHelp, parseCliArgs, readPackageVersion } from './cli.js';

describe('parseCliArgs', () => {
  it('defaults to server when argv is empty', () => {
    assert.equal(parseCliArgs([]), 'server');
  });

  it('treats unknown flags as server so MCP clients can pass extra args', () => {
    assert.equal(parseCliArgs(['--stdio']), 'server');
  });

  it('recognises --help and -h', () => {
    assert.equal(parseCliArgs(['--help']), 'help');
    assert.equal(parseCliArgs(['-h']), 'help');
  });

  it('recognises --version and -v', () => {
    assert.equal(parseCliArgs(['--version']), 'version');
    assert.equal(parseCliArgs(['-v']), 'version');
  });

  it('prefers help when both help and version are present', () => {
    assert.equal(parseCliArgs(['--version', '--help']), 'help');
  });
});

describe('readPackageVersion', () => {
  it('reads the workspace package.json version', () => {
    const v = readPackageVersion();
    assert.match(v, /^\d+\.\d+\.\d+$/);
    assert.notEqual(v, '0.0.0');
  });
});

describe('consumeDbPath', () => {
  it('reads --path and -p', () => {
    assert.equal(consumeDbPath(['--path', '/tmp/db']), '/tmp/db');
    assert.equal(consumeDbPath(['-p', '/tmp/db2']), '/tmp/db2');
    assert.equal(consumeDbPath(['--path=/tmp/db3']), '/tmp/db3');
    assert.equal(consumeDbPath([]), undefined);
  });
});

describe('formatHelp', () => {
  it('includes the version, usage, and npm URL', () => {
    const text = formatHelp('1.1.0');
    assert.match(text, /logosdb-mcp-server 1\.1\.0/);
    assert.match(text, /--help/);
    assert.match(text, /--version/);
    assert.match(text, /--path/);
    assert.match(text, /npmjs\.com\/package\/logosdb-mcp-server/);
  });
});

describe('applyCli', () => {
  it('sets LOGOSDB_PATH from --path', () => {
    const prev = process.env.LOGOSDB_PATH;
    try {
      applyCli(['--path', '/tmp/logosdb-cli-test'], {
        stdout: { write: () => true } as unknown as NodeJS.WritableStream,
      });
      assert.equal(process.env.LOGOSDB_PATH, '/tmp/logosdb-cli-test');
    } finally {
      if (prev === undefined) delete process.env.LOGOSDB_PATH;
      else process.env.LOGOSDB_PATH = prev;
    }
  });

  it('writes version and does not start the server', () => {
    let out = '';
    const mode = applyCli(['--version'], {
      stdout: { write: (s: string) => { out += s; return true; } } as NodeJS.WritableStream,
    });
    assert.equal(mode, 'version');
    assert.match(out, /^\d+\.\d+\.\d+\n$/);
  });

  it('writes help text', () => {
    let out = '';
    const mode = applyCli(['-h'], {
      stdout: { write: (s: string) => { out += s; return true; } } as NodeJS.WritableStream,
    });
    assert.equal(mode, 'help');
    assert.match(out, /Usage:/);
  });
});
