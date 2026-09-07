// Minimal ambient types for Node's built-in SQLite module.
//
// node:sqlite ships with the Node runtime (22+), but typings for it only landed
// in @types/node 22, and this project pins ^20. Declaring the small surface we
// actually use is less disruptive than bumping @types/node across a codebase
// that currently typechecks clean — swap this file for the real typings when the
// dependency is upgraded.
declare module "node:sqlite" {
  type SQLValue = string | number | bigint | null | Uint8Array;

  interface StatementSync {
    run(...params: SQLValue[]): { changes: number; lastInsertRowid: number | bigint };
    get(...params: SQLValue[]): unknown;
    all(...params: SQLValue[]): unknown[];
  }

  export class DatabaseSync {
    constructor(path: string, options?: { open?: boolean; readOnly?: boolean });
    exec(sql: string): void;
    prepare(sql: string): StatementSync;
    close(): void;
  }
}
