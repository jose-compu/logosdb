/**
 * Lightweight argv handling for `npx logosdb-mcp-server --help|--version`.
 * MCP stdio mode is the default when no CLI flag is present.
 */

export type CliMode = 'help' | 'version' | 'server';

export const PACKAGE_NAME = 'logosdb-mcp-server';

/** Optional `--path` / `-p` override for LOGOSDB_PATH. */
export function consumeDbPath(argv: string[]): string | undefined {
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--path' || arg === '-p') {
      const value = argv[i + 1];
      if (value && !value.startsWith('-')) return value;
    }
    if (arg.startsWith('--path=')) {
      const value = arg.slice('--path='.length);
      if (value) return value;
    }
  }
  return undefined;
}

export function readPackageVersion(): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const pkg = require('../package.json') as { version?: string };
    if (pkg?.version) return pkg.version;
  } catch {
    /* published layout or tests without a sibling package.json */
  }
  return '0.0.0';
}

export function parseCliArgs(argv: string[]): CliMode {
  if (argv.some((arg) => arg === '--help' || arg === '-h')) return 'help';
  if (argv.some((arg) => arg === '--version' || arg === '-v')) return 'version';
  return 'server';
}

export function formatHelp(version: string): string {
  return `${PACKAGE_NAME} ${version}

MCP server exposing LogosDB semantic search over stdio.

Usage:
  logosdb-mcp-server              Start the MCP server (stdio)
  logosdb-mcp-server --path DIR   Start with LOGOSDB_PATH=DIR
  logosdb-mcp-server --help       Show this help
  logosdb-mcp-server --version    Print the package version

Typical Claude Code / Cursor config:
  command: npx
  args: ["-y", "logosdb-mcp-server"]
  env: { "LOGOSDB_PATH": "./.logosdb" }

Docs: https://www.npmjs.com/package/logosdb-mcp-server
`;
}

export function applyCli(argv: string[], io: { stdout: NodeJS.WritableStream } = process): CliMode {
  const dbPath = consumeDbPath(argv);
  if (dbPath) process.env.LOGOSDB_PATH = dbPath;
  const mode = parseCliArgs(argv);
  const version = readPackageVersion();
  if (mode === 'help') {
    io.stdout.write(formatHelp(version));
  } else if (mode === 'version') {
    io.stdout.write(`${version}\n`);
  }
  return mode;
}
