/**
 * The snapshot payload (WP-P3): table cells keep their anchors within the
 * node budget, zero-size leaves are kept only with an anchor, and a
 * snapshot sends the interacted element's anchor chain and a text digest
 * instead of the viewport tree.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  anchorChain,
  buildDigest,
  captureSnapshot,
  noteInteraction,
  truncateTable,
  walkDOM,
  MAX_DIGEST_CHARS,
  MAX_NODE_COUNT,
} from '../src/dom-capture';
import type { NodeCounter } from '../src/dom-capture';
import type { DomNode } from '../src/types';

function mockBBox(el: Element, width = 100, height = 20, y = 0): void {
  const rect = {
    x: 0, y, width, height, top: y, left: 0, bottom: y + height, right: width,
    toJSON() { return this; },
  };
  el.getBoundingClientRect = () => rect as DOMRect;
}

function flatten(nodes: DomNode[]): DomNode[] {
  const out: DomNode[] = [];
  const visit = (n: DomNode): void => {
    out.push(n);
    (n.children ?? []).forEach(visit);
  };
  nodes.forEach(visit);
  return out;
}

/** A `rows`-row table whose cells hold a customer link and an Approve button. */
function invoiceTable(rows: number): HTMLTableElement {
  const table = document.createElement('table');
  const thead = table.createTHead();
  const head = thead.insertRow();
  for (const label of ['Customer', 'Action']) {
    const th = document.createElement('th');
    th.textContent = label;
    head.appendChild(th);
  }
  const tbody = table.createTBody();
  for (let i = 0; i < rows; i++) {
    const row = tbody.insertRow();
    const name = row.insertCell();
    const link = document.createElement('a');
    link.href = `/customers/${i}`;
    link.textContent = `Customer ${i}`;
    name.appendChild(link);
    const action = row.insertCell();
    const button = document.createElement('button');
    button.setAttribute('aria-label', `Approve invoice ${i}`);
    button.textContent = 'Approve';
    action.appendChild(button);
    for (const el of [row, name, link, action, button]) mockBBox(el, 100, 20, i * 20);
  }
  document.body.appendChild(table);
  return table;
}

beforeEach(() => {
  document.body.innerHTML = '';
  Object.defineProperty(window, 'innerWidth', { value: 1024, configurable: true });
  Object.defineProperty(window, 'innerHeight', { value: 100_000, configurable: true });
  noteInteraction(null);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('truncateTable (#118)', () => {
  it('keeps the links and buttons inside cells', () => {
    const result = truncateTable(invoiceTable(3));
    const nodes = flatten([result]);
    const link = nodes.find((n) => n.tag === 'a' && n.innerText === 'Customer 1');
    const button = nodes.find((n) => n.ariaLabel === 'Approve invoice 1');
    expect(link).toBeDefined();
    expect(button).toBeDefined();
    expect(button!.tag).toBe('button');
  });

  it('respects the node budget on a 500-row table and flags the cap', () => {
    vi.spyOn(window, 'getComputedStyle').mockReturnValue(
      { display: 'block', visibility: 'visible' } as CSSStyleDeclaration,
    );
    const table = invoiceTable(500);
    const counter: NodeCounter = { count: MAX_NODE_COUNT - 40, truncated: false };
    const result = walkDOM(table, 0, counter);
    expect(result).not.toBeNull();
    const emitted = flatten([result!]).filter((n) => n.tag !== '#truncated');
    expect(counter.count).toBeLessThanOrEqual(MAX_NODE_COUNT);
    expect(emitted.length).toBeLessThanOrEqual(40);
    expect(counter.truncated).toBe(true);
    // The rows that made it in still carry their anchors.
    expect(emitted.some((n) => n.tag === 'a' && n.innerText === 'Customer 0')).toBe(true);
  });

  it('applies maxRows to head and foot rows too', () => {
    const table = document.createElement('table');
    const tfoot = table.createTFoot();
    for (let i = 0; i < 5; i++) tfoot.insertRow().insertCell().textContent = `Total ${i}`;
    document.body.appendChild(table);
    const result = truncateTable(table, 2);
    const foot = result.children!.find((c) => c.tag === 'tfoot');
    expect(foot!.children!.length).toBe(2);
  });
});

describe('zero-size leaves (#119)', () => {
  it('keeps a 0x0 display:contents span carrying text and drops an empty 0x0 div', () => {
    const host = document.createElement('div');
    const span = document.createElement('span');
    span.style.display = 'contents';
    span.textContent = 'Invoice #4411';
    const empty = document.createElement('div');
    host.append(span, empty);
    document.body.appendChild(host);
    mockBBox(host, 300, 40);
    mockBBox(span, 0, 0);
    mockBBox(empty, 0, 0);

    const result = walkDOM(host);
    expect(result!.children!.length).toBe(1);
    expect(result!.children![0].innerText).toBe('Invoice #4411');
  });
});

describe('anchor set', () => {
  it('records the interacted element and its ancestors up to the landmark', () => {
    document.body.innerHTML = `
      <div class="app">
        <nav aria-label="Main">
          <ul><li class="item"><a href="/inbox" data-testid="nav-inbox"><span>Inbox</span></a></li></ul>
        </nav>
      </div>`;
    const span = document.querySelector('span')!;
    const chain = anchorChain(span);
    expect(chain.map((a) => a.tag)).toEqual(['span', 'a', 'li', 'ul', 'nav']);
    expect(chain[0].interacted).toBe(true);
    expect(chain[0].selector).toBe('nav > ul > li.item > a > span');
    const link = chain[1];
    expect(link.testId).toBe('nav-inbox');
    expect(link.name).toBe('Inbox');
    expect(link.href).toMatch(/\/inbox$/);
    expect(chain[4].name).toBe('Main');
  });

  it('names a field from its label, never from its value', () => {
    document.body.innerHTML = `
      <form><label for="email">Work email</label><input id="email" type="email" value="jane@example.com"></form>`;
    const input = document.querySelector('input')!;
    const [record] = anchorChain(input);
    expect(record).toMatchObject({ tag: 'input', type: 'email', id: 'email', name: 'Work email' });
    expect(JSON.stringify(record)).not.toContain('jane@example.com');
  });

  it('adds the typed value only with captureInputValues, and never a password', () => {
    document.body.innerHTML = `<form><input id="q" value="quarterly report"><input id="p" type="password" value="hunter2"></form>`;
    const [plain] = anchorChain(document.getElementById('q')!, { captureInputValues: true });
    expect(plain.value).toBe('quarterly report');
    const [secret] = anchorChain(document.getElementById('p')!, { captureInputValues: true });
    expect(secret.value).toBeUndefined();
    const [off] = anchorChain(document.getElementById('q')!);
    expect(off.value).toBeUndefined();
  });

  it('sends an interaction once, then forgets it', () => {
    document.body.innerHTML = `<main><button aria-label="Send">Send</button></main>`;
    noteInteraction(document.querySelector('button'));
    const first = captureSnapshot();
    expect(first.anchors[0]).toMatchObject({ tag: 'button', ariaLabel: 'Send', interacted: true });
    expect(first.anchors[first.anchors.length - 1].tag).toBe('main');
    expect(captureSnapshot().anchors).toEqual([]);
  });

  it('digests headings and list-row labels', () => {
    const tree: DomNode[] = [
      { tag: 'h1', innerText: 'Hacker News' },
      {
        tag: 'table', children: [
          { tag: 'tr', children: [{ tag: 'span', innerText: '1.' }, { tag: 'a', innerText: 'The quiet death of the RSS reader' }] },
          { tag: 'tr', children: [{ tag: 'td', innerText: '530 points' }] },
        ],
      },
      { tag: 'ul', children: [{ tag: 'li', innerText: 'Settings' }] },
    ];
    expect(buildDigest(tree)).toBe('Hacker News\nThe quiet death of the RSS reader\n530 points\nSettings');
  });

  it('keeps a snapshot of a 5,000-node page under 8 KB', () => {
    vi.spyOn(window, 'getComputedStyle').mockReturnValue(
      { display: 'block', visibility: 'visible' } as CSSStyleDeclaration,
    );
    const main = document.createElement('main');
    const list = document.createElement('ul');
    main.appendChild(list);
    let target: Element | null = null;
    for (let i = 0; i < 2500; i++) {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = `https://example.com/records/${i}?view=full`;
      a.textContent = `Record ${i}: a long enough row label to fill the digest quickly`;
      li.appendChild(a);
      list.appendChild(li);
      if (i === 1200) target = a;
    }
    document.body.appendChild(main);
    noteInteraction(target);

    const snapshot = captureSnapshot();
    const payload = JSON.stringify({
      ...snapshot,
      url: 'https://example.com/records',
      tabId: 7,
      captureReason: 'request',
    });
    expect(new TextEncoder().encode(payload).length).toBeLessThan(8 * 1024);
    expect(snapshot.digest.length).toBeLessThanOrEqual(MAX_DIGEST_CHARS);
    expect(snapshot.anchors[0]).toMatchObject({
      tag: 'a', interacted: true, href: 'https://example.com/records/1200',
    });
    expect(snapshot.anchors[0].name).toMatch(/^Record 1200:/);
    expect(snapshot.anchors[snapshot.anchors.length - 1].tag).toBe('main');
  }, 30_000);
});
