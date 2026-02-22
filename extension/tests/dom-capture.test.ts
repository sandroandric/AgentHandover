/**
 * Tests for the viewport-bounded DOM snapshot capture module.
 *
 * Tests cover:
 *   1. CSS rot class stripping (Emotion, styled-components, CSS Modules, generic hash)
 *   2. Text truncation (word boundary, hard limit, short text passthrough)
 *   3. Table row truncation (visible rows, head/foot preservation, truncation marker)
 *   4. Viewport intersection logic (mocked getBoundingClientRect)
 *   5. Semantic info extraction (ARIA, data-testid, bbox, state)
 *   6. Full DOM capture integration (walkDOM, captureViewportDOM)
 *
 * Environment: jsdom via vitest (configured in vitest.config.ts).
 *
 * NOTE: Full cross-component integration testing (Extension -> Daemon -> Storage -> Worker)
 * requires browser automation (e.g., Puppeteer) and is not possible in jsdom.
 * See tests/integration/ for Python-based pipeline integration tests.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  stripCssRot,
  isCssRot,
  truncateText,
  truncateTable,
  isInViewport,
  extractNodeInfo,
  walkDOM,
  captureViewportDOM,
  MAX_TABLE_ROWS,
  MAX_TEXT_LENGTH,
  MAX_TREE_DEPTH,
  MAX_NODE_COUNT,
} from '../src/dom-capture';
import type { NodeCounter } from '../src/dom-capture';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Mocks getBoundingClientRect on an element to return the given rect values.
 */
function mockBBox(
  el: Element,
  rect: { x?: number; y?: number; width?: number; height?: number },
): void {
  const full = {
    x: rect.x ?? 0,
    y: rect.y ?? 0,
    width: rect.width ?? 100,
    height: rect.height ?? 50,
    top: rect.y ?? 0,
    left: rect.x ?? 0,
    bottom: (rect.y ?? 0) + (rect.height ?? 50),
    right: (rect.x ?? 0) + (rect.width ?? 100),
    toJSON() { return this; },
  };
  el.getBoundingClientRect = () => full;
}

/**
 * Sets window.innerWidth and window.innerHeight for viewport intersection tests.
 */
function setViewportSize(width: number, height: number): void {
  Object.defineProperty(window, 'innerWidth', { value: width, writable: true, configurable: true });
  Object.defineProperty(window, 'innerHeight', { value: height, writable: true, configurable: true });
}

// ---------------------------------------------------------------------------
// CSS Rot Stripping
// ---------------------------------------------------------------------------

describe('stripCssRot', () => {
  it('removes Emotion classes (css-...)', () => {
    const result = stripCssRot(['css-1a2b3c', 'my-button', 'css-0']);
    expect(result).toEqual(['my-button']);
  });

  it('removes styled-components classes (sc-...)', () => {
    const result = stripCssRot(['sc-bdfBjE', 'header', 'sc-abc123']);
    expect(result).toEqual(['header']);
  });

  it('removes CSS Modules classes (module_component__hash)', () => {
    const result = stripCssRot(['styles_container__a1b2c', 'app-header']);
    expect(result).toEqual(['app-header']);
  });

  it('removes generic hash-based classes (prefix-hash)', () => {
    const result = stripCssRot(['a-b1c2d3e4', 'xy-ab12cd34ef', 'main-nav']);
    expect(result).toEqual(['main-nav']);
  });

  it('preserves Tailwind utility classes', () => {
    const tailwind = ['flex', 'items-center', 'px-4', 'py-2', 'bg-blue-500', 'text-white'];
    const result = stripCssRot(tailwind);
    expect(result).toEqual(tailwind);
  });

  it('preserves BEM-style classes', () => {
    const bem = ['header__nav', 'header__nav--active', 'btn--primary'];
    const result = stripCssRot(bem);
    expect(result).toEqual(bem);
  });

  it('preserves semantic class names', () => {
    const semantic = ['container', 'sidebar', 'main-content', 'footer-links'];
    const result = stripCssRot(semantic);
    expect(result).toEqual(semantic);
  });

  it('returns empty array when all classes are rot', () => {
    const result = stripCssRot(['css-abc', 'sc-XYZ123', 'a-b1c2d3e4']);
    expect(result).toEqual([]);
  });

  it('returns empty array for empty input', () => {
    const result = stripCssRot([]);
    expect(result).toEqual([]);
  });
});

describe('isCssRot', () => {
  it('returns true for Emotion classes', () => {
    expect(isCssRot('css-1a2b3c')).toBe(true);
    expect(isCssRot('css-0')).toBe(true);
  });

  it('returns true for styled-components classes', () => {
    expect(isCssRot('sc-dqNmKP')).toBe(true);
    expect(isCssRot('sc-abc123')).toBe(true);
  });

  it('returns true for CSS Modules classes', () => {
    expect(isCssRot('styles_container__a1b2c')).toBe(true);
  });

  it('returns true for generic hash classes', () => {
    expect(isCssRot('ab-c1d2e3f4')).toBe(true);
  });

  it('returns false for normal classes', () => {
    expect(isCssRot('button')).toBe(false);
    expect(isCssRot('main-navigation')).toBe(false);
    expect(isCssRot('flex')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Text Truncation
// ---------------------------------------------------------------------------

describe('truncateText', () => {
  it('returns short text unchanged', () => {
    expect(truncateText('Hello world')).toBe('Hello world');
  });

  it('returns text at exactly max length unchanged', () => {
    const exact = 'x'.repeat(MAX_TEXT_LENGTH);
    expect(truncateText(exact)).toBe(exact);
  });

  it('truncates text longer than max length with ellipsis', () => {
    const long = 'x'.repeat(MAX_TEXT_LENGTH + 100);
    const result = truncateText(long);
    expect(result.endsWith('...')).toBe(true);
    // The result (excluding ellipsis) should be at most MAX_TEXT_LENGTH chars.
    expect(result.length).toBeLessThanOrEqual(MAX_TEXT_LENGTH + 3);
  });

  it('breaks at word boundary when possible', () => {
    // Create text where a space appears near the end, within the 80% window.
    const words = 'The quick brown fox jumps over the lazy dog. ';
    const repeated = words.repeat(20); // Much longer than 500 chars.
    const result = truncateText(repeated, 50);
    expect(result.endsWith('...')).toBe(true);
    // The text before the ellipsis should be shorter than the hard limit,
    // because we broke at a word boundary (space) instead of mid-word.
    const beforeEllipsis = result.slice(0, -3);
    expect(beforeEllipsis.length).toBeLessThanOrEqual(50);
    // It should end at a complete word — the last character before '...'
    // should not be in the middle of a word. We verify by checking that
    // no partial word was cut (the char at position length is a space in
    // the original string, i.e., we broke at a word boundary).
    const nextCharInOriginal = repeated[beforeEllipsis.length];
    expect(nextCharInOriginal).toBe(' ');
  });

  it('uses hard limit when no word boundary is near', () => {
    const noSpaces = 'a'.repeat(600);
    const result = truncateText(noSpaces, 100);
    expect(result).toBe('a'.repeat(100) + '...');
  });

  it('respects custom maxChars parameter', () => {
    const result = truncateText('Hello beautiful world', 10);
    expect(result.length).toBeLessThanOrEqual(13); // 10 + '...'
  });

  it('handles empty string', () => {
    expect(truncateText('')).toBe('');
  });
});

// ---------------------------------------------------------------------------
// Viewport Intersection
// ---------------------------------------------------------------------------

describe('isInViewport', () => {
  beforeEach(() => {
    setViewportSize(1024, 768);
  });

  it('returns true for element fully inside viewport', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: 100, y: 100, width: 200, height: 100 });
    expect(isInViewport(el)).toBe(true);
  });

  it('returns true for element partially visible (top edge clipped)', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: 100, y: -20, width: 200, height: 100 });
    expect(isInViewport(el)).toBe(true);
  });

  it('returns true for element partially visible (left edge clipped)', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: -50, y: 100, width: 200, height: 100 });
    expect(isInViewport(el)).toBe(true);
  });

  it('returns true for element partially visible (bottom edge clipped)', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: 100, y: 700, width: 200, height: 100 });
    expect(isInViewport(el)).toBe(true);
  });

  it('returns true for element partially visible (right edge clipped)', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: 900, y: 100, width: 200, height: 100 });
    expect(isInViewport(el)).toBe(true);
  });

  it('returns false for element completely above viewport', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: 100, y: -200, width: 200, height: 100 });
    expect(isInViewport(el)).toBe(false);
  });

  it('returns false for element completely below viewport', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: 100, y: 900, width: 200, height: 100 });
    expect(isInViewport(el)).toBe(false);
  });

  it('returns false for element completely to the left of viewport', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: -300, y: 100, width: 200, height: 100 });
    expect(isInViewport(el)).toBe(false);
  });

  it('returns false for element completely to the right of viewport', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: 1200, y: 100, width: 200, height: 100 });
    expect(isInViewport(el)).toBe(false);
  });

  it('returns false for zero-width element', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: 100, y: 100, width: 0, height: 100 });
    expect(isInViewport(el)).toBe(false);
  });

  it('returns false for zero-height element', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: 100, y: 100, width: 200, height: 0 });
    expect(isInViewport(el)).toBe(false);
  });

  it('returns true for element spanning entire viewport', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: -50, y: -50, width: 1200, height: 900 });
    expect(isInViewport(el)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Semantic Info Extraction
// ---------------------------------------------------------------------------

describe('extractNodeInfo', () => {
  beforeEach(() => {
    setViewportSize(1024, 768);
  });

  it('extracts tag name in lowercase', () => {
    const el = document.createElement('div');
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.tag).toBe('div');
  });

  it('extracts ARIA role', () => {
    const el = document.createElement('div');
    el.setAttribute('role', 'navigation');
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.role).toBe('navigation');
  });

  it('extracts aria-label', () => {
    const el = document.createElement('button');
    el.setAttribute('aria-label', 'Close dialog');
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.ariaLabel).toBe('Close dialog');
  });

  it('extracts data-testid', () => {
    const el = document.createElement('div');
    el.setAttribute('data-testid', 'login-form');
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.testId).toBe('login-form');
  });

  it('extracts data-test-id (alternative naming)', () => {
    const el = document.createElement('div');
    el.setAttribute('data-test-id', 'signup-btn');
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.testId).toBe('signup-btn');
  });

  it('extracts data-cy (Cypress test attribute)', () => {
    const el = document.createElement('div');
    el.setAttribute('data-cy', 'submit');
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.testId).toBe('submit');
  });

  it('extracts bounding box with rounded values', () => {
    const el = document.createElement('div');
    mockBBox(el, { x: 10.7, y: 20.3, width: 150.9, height: 75.1 });
    const info = extractNodeInfo(el);
    expect(info.bbox).toEqual({ x: 11, y: 20, width: 151, height: 75 });
  });

  it('extracts disabled state for input', () => {
    const el = document.createElement('input');
    el.disabled = true;
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.state?.enabled).toBe(false);
  });

  it('extracts checked state for checkbox', () => {
    const el = document.createElement('input');
    el.type = 'checkbox';
    el.checked = true;
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.state?.checked).toBe(true);
    expect(info.tag).toBe('input[type=checkbox]');
  });

  it('extracts placeholder as name when no aria-label', () => {
    const el = document.createElement('input');
    el.placeholder = 'Enter your email';
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.name).toBe('Enter your email');
  });

  it('does not capture password field value', () => {
    const el = document.createElement('input');
    el.type = 'password';
    el.value = 'supersecret123';
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.innerText).toBeUndefined();
  });

  it('extracts select element selected option', () => {
    const el = document.createElement('select');
    const opt1 = document.createElement('option');
    opt1.text = 'Option A';
    opt1.value = 'a';
    const opt2 = document.createElement('option');
    opt2.text = 'Option B';
    opt2.value = 'b';
    opt2.selected = true;
    el.appendChild(opt1);
    el.appendChild(opt2);
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.innerText).toBe('Option B');
    expect(info.state?.selected).toBe(true);
  });

  it('extracts aria-checked for custom widgets', () => {
    const el = document.createElement('div');
    el.setAttribute('role', 'checkbox');
    el.setAttribute('aria-checked', 'true');
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.state?.checked).toBe(true);
  });

  it('extracts aria-disabled for custom widgets', () => {
    const el = document.createElement('div');
    el.setAttribute('role', 'button');
    el.setAttribute('aria-disabled', 'true');
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.state?.enabled).toBe(false);
  });

  it('strips CSS rot classes from name', () => {
    const el = document.createElement('div');
    el.classList.add('css-1a2b3c', 'main-container', 'sc-bdfBjE');
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    // Only the meaningful class survives.
    expect(info.name).toBe('main-container');
  });

  it('resolves aria-labelledby to text', () => {
    const label = document.createElement('span');
    label.id = 'my-label';
    label.textContent = 'Username field';
    document.body.appendChild(label);

    const el = document.createElement('input');
    el.setAttribute('aria-labelledby', 'my-label');
    mockBBox(el, {});
    const info = extractNodeInfo(el);
    expect(info.name).toBe('Username field');

    document.body.removeChild(label);
  });
});

// ---------------------------------------------------------------------------
// Table Truncation
// ---------------------------------------------------------------------------

describe('truncateTable', () => {
  beforeEach(() => {
    setViewportSize(1024, 768);
  });

  it('preserves thead rows', () => {
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    const th = document.createElement('th');
    th.textContent = 'Name';
    headRow.appendChild(th);
    thead.appendChild(headRow);
    table.appendChild(thead);

    // Add a tbody with a single visible row.
    const tbody = document.createElement('tbody');
    const row = document.createElement('tr');
    const td = document.createElement('td');
    td.textContent = 'Alice';
    row.appendChild(td);
    tbody.appendChild(row);
    table.appendChild(tbody);

    document.body.appendChild(table);

    // Mock all elements as in-viewport.
    mockBBox(table, { x: 0, y: 0, width: 500, height: 200 });
    mockBBox(thead, { x: 0, y: 0, width: 500, height: 30 });
    mockBBox(headRow, { x: 0, y: 0, width: 500, height: 30 });
    mockBBox(th, { x: 0, y: 0, width: 500, height: 30 });
    mockBBox(tbody, { x: 0, y: 30, width: 500, height: 170 });
    mockBBox(row, { x: 0, y: 30, width: 500, height: 30 });
    mockBBox(td, { x: 0, y: 30, width: 500, height: 30 });

    const result = truncateTable(table);
    expect(result.tag).toBe('table');
    expect(result.children).toBeDefined();

    // thead should be present.
    const theadNode = result.children!.find((c) => c.tag === 'thead');
    expect(theadNode).toBeDefined();
    expect(theadNode!.children).toBeDefined();
    expect(theadNode!.children!.length).toBe(1);

    document.body.removeChild(table);
  });

  it('truncates body rows beyond maxRows and adds marker', () => {
    const table = document.createElement('table');
    const tbody = document.createElement('tbody');

    // Add 100 rows, all "in viewport".
    for (let i = 0; i < 100; i++) {
      const row = document.createElement('tr');
      const td = document.createElement('td');
      td.textContent = `Row ${i}`;
      row.appendChild(td);
      tbody.appendChild(row);

      // All rows are visible in viewport.
      mockBBox(row, { x: 0, y: i * 20, width: 500, height: 20 });
      mockBBox(td, { x: 0, y: i * 20, width: 500, height: 20 });
    }
    table.appendChild(tbody);
    document.body.appendChild(table);

    mockBBox(table, { x: 0, y: 0, width: 500, height: 2000 });
    mockBBox(tbody, { x: 0, y: 0, width: 500, height: 2000 });

    const result = truncateTable(table, 10);
    expect(result.tag).toBe('table');

    // tbody should be present.
    const tbodyNode = result.children!.find((c) => c.tag === 'tbody');
    expect(tbodyNode).toBeDefined();

    // Should have exactly 10 data rows.
    expect(tbodyNode!.children!.length).toBe(10);

    // A truncation marker should be present.
    const marker = result.children!.find((c) => c.tag === '#truncated');
    expect(marker).toBeDefined();
    expect(marker!.innerText).toContain('90 more rows truncated');

    document.body.removeChild(table);
  });

  it('does not add truncation marker when all rows fit', () => {
    const table = document.createElement('table');
    const tbody = document.createElement('tbody');

    for (let i = 0; i < 5; i++) {
      const row = document.createElement('tr');
      const td = document.createElement('td');
      td.textContent = `Row ${i}`;
      row.appendChild(td);
      tbody.appendChild(row);
      mockBBox(row, { x: 0, y: i * 20, width: 500, height: 20 });
      mockBBox(td, { x: 0, y: i * 20, width: 500, height: 20 });
    }
    table.appendChild(tbody);
    document.body.appendChild(table);

    mockBBox(table, { x: 0, y: 0, width: 500, height: 100 });
    mockBBox(tbody, { x: 0, y: 0, width: 500, height: 100 });

    const result = truncateTable(table, 50);
    const marker = result.children!.find((c) => c.tag === '#truncated');
    expect(marker).toBeUndefined();

    document.body.removeChild(table);
  });
});

// ---------------------------------------------------------------------------
// DOM Tree Walker (walkDOM)
// ---------------------------------------------------------------------------

describe('walkDOM', () => {
  beforeEach(() => {
    setViewportSize(1024, 768);
    document.body.innerHTML = '';
  });

  it('returns null for text nodes', () => {
    const text = document.createTextNode('hello');
    const result = walkDOM(text);
    expect(result).toBeNull();
  });

  it('returns null for script elements', () => {
    const script = document.createElement('script');
    script.textContent = 'console.log("test")';
    const result = walkDOM(script);
    expect(result).toBeNull();
  });

  it('returns null for style elements', () => {
    const style = document.createElement('style');
    style.textContent = 'body { color: red; }';
    const result = walkDOM(style);
    expect(result).toBeNull();
  });

  it('returns null for hidden elements', () => {
    const el = document.createElement('div');
    el.setAttribute('hidden', '');
    mockBBox(el, { x: 0, y: 0, width: 100, height: 100 });
    const result = walkDOM(el);
    expect(result).toBeNull();
  });

  it('returns null for elements outside viewport', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    mockBBox(el, { x: 2000, y: 2000, width: 100, height: 100 });
    const result = walkDOM(el);
    expect(result).toBeNull();
  });

  it('captures a simple visible div with text', () => {
    const el = document.createElement('p');
    el.textContent = 'Hello world';
    document.body.appendChild(el);
    mockBBox(el, { x: 10, y: 10, width: 200, height: 30 });

    const result = walkDOM(el);
    expect(result).not.toBeNull();
    expect(result!.tag).toBe('p');
    expect(result!.innerText).toBe('Hello world');
  });

  it('captures nested elements', () => {
    const parent = document.createElement('div');
    const child = document.createElement('span');
    child.textContent = 'nested text';
    parent.appendChild(child);
    document.body.appendChild(parent);

    mockBBox(parent, { x: 10, y: 10, width: 200, height: 50 });
    mockBBox(child, { x: 15, y: 15, width: 100, height: 20 });

    const result = walkDOM(parent);
    expect(result).not.toBeNull();
    expect(result!.tag).toBe('div');
    expect(result!.children).toBeDefined();
    expect(result!.children!.length).toBe(1);
    expect(result!.children![0].tag).toBe('span');
    expect(result!.children![0].innerText).toBe('nested text');
  });

  it('respects MAX_TREE_DEPTH', () => {
    // Build a chain deeper than MAX_TREE_DEPTH.
    let current = document.createElement('div');
    document.body.appendChild(current);
    mockBBox(current, { x: 0, y: 0, width: 100, height: 100 });

    const root = current;
    for (let i = 0; i < MAX_TREE_DEPTH + 5; i++) {
      const child = document.createElement('div');
      child.setAttribute('data-depth', String(i));
      current.appendChild(child);
      mockBBox(child, { x: 0, y: 0, width: 100, height: 100 });
      current = child;
    }

    const result = walkDOM(root, 0);
    expect(result).not.toBeNull();

    // Walk down the tree and verify it stops at MAX_TREE_DEPTH.
    let node = result;
    let depth = 0;
    while (node?.children && node.children.length > 0) {
      node = node.children[0];
      depth++;
    }
    expect(depth).toBeLessThanOrEqual(MAX_TREE_DEPTH);
  });

  it('captures table elements with truncation', () => {
    const table = document.createElement('table');
    const tbody = document.createElement('tbody');
    for (let i = 0; i < 5; i++) {
      const row = document.createElement('tr');
      const td = document.createElement('td');
      td.textContent = `Cell ${i}`;
      row.appendChild(td);
      tbody.appendChild(row);
      mockBBox(row, { x: 0, y: i * 20, width: 500, height: 20 });
      mockBBox(td, { x: 0, y: i * 20, width: 500, height: 20 });
    }
    table.appendChild(tbody);
    document.body.appendChild(table);

    mockBBox(table, { x: 0, y: 0, width: 500, height: 100 });
    mockBBox(tbody, { x: 0, y: 0, width: 500, height: 100 });

    const result = walkDOM(table);
    expect(result).not.toBeNull();
    expect(result!.tag).toBe('table');
  });
});

// ---------------------------------------------------------------------------
// Node Count Cap (MAX_NODE_COUNT)
// ---------------------------------------------------------------------------

describe('MAX_NODE_COUNT cap', () => {
  beforeEach(() => {
    setViewportSize(1024, 768);
    document.body.innerHTML = '';
  });

  it('walkDOM respects counter and stops producing nodes', () => {
    // Build a flat list of children under a parent.
    const parent = document.createElement('div');
    document.body.appendChild(parent);
    mockBBox(parent, { x: 0, y: 0, width: 800, height: 600 });

    for (let i = 0; i < 20; i++) {
      const child = document.createElement('span');
      child.textContent = `Item ${i}`;
      parent.appendChild(child);
      mockBBox(child, { x: 0, y: i * 20, width: 100, height: 18 });
    }

    // Pass a counter that already has a high count, leaving room for only a few nodes.
    const counter: NodeCounter = { count: MAX_NODE_COUNT - 5, truncated: false };
    const result = walkDOM(parent, 0, counter);

    expect(result).not.toBeNull();
    // The parent itself counts as 1, so at most 4 children can be produced.
    // The counter should have been set to truncated.
    expect(counter.truncated).toBe(true);
    // Children produced should be fewer than the 20 available.
    const childCount = result!.children?.length ?? 0;
    expect(childCount).toBeLessThan(20);
  });

  it('walkDOM returns null when counter is already at max', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    mockBBox(el, { x: 10, y: 10, width: 100, height: 50 });

    const counter: NodeCounter = { count: MAX_NODE_COUNT, truncated: false };
    const result = walkDOM(el, 0, counter);

    expect(result).toBeNull();
    expect(counter.truncated).toBe(true);
  });

  it('captureViewportDOM appends #truncated marker when cap is hit', () => {
    // Create more elements than MAX_NODE_COUNT.
    // Each element is a leaf, so each produces exactly 1 node.
    const count = MAX_NODE_COUNT + 100;
    for (let i = 0; i < count; i++) {
      const el = document.createElement('span');
      el.textContent = `N${i}`;
      document.body.appendChild(el);
      // Place them in viewport.
      mockBBox(el, { x: 0, y: 0, width: 50, height: 10 });
    }

    const result = captureViewportDOM();

    // The last element should be the truncation marker.
    const lastNode = result[result.length - 1];
    expect(lastNode.tag).toBe('#truncated');
    expect(lastNode.innerText).toContain(`${MAX_NODE_COUNT}`);

    // Total real nodes (excluding marker) should be at most MAX_NODE_COUNT.
    const realNodes = result.filter((n) => n.tag !== '#truncated');
    expect(realNodes.length).toBeLessThanOrEqual(MAX_NODE_COUNT);
  });

  it('captureViewportDOM does not add #truncated when under cap', () => {
    // Create just a few elements — well under the cap.
    for (let i = 0; i < 5; i++) {
      const el = document.createElement('div');
      el.setAttribute('data-testid', `el-${i}`);
      document.body.appendChild(el);
      mockBBox(el, { x: 0, y: i * 50, width: 200, height: 40 });
    }

    const result = captureViewportDOM();
    const marker = result.find((n) => n.tag === '#truncated');
    expect(marker).toBeUndefined();
    expect(result.length).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// Full Capture Integration
// ---------------------------------------------------------------------------

describe('captureViewportDOM', () => {
  beforeEach(() => {
    setViewportSize(1024, 768);
    document.body.innerHTML = '';
  });

  it('returns empty array when body has no children', () => {
    const result = captureViewportDOM();
    expect(result).toEqual([]);
  });

  it('captures visible elements and skips off-screen ones', () => {
    const visible = document.createElement('div');
    visible.setAttribute('data-testid', 'visible');
    document.body.appendChild(visible);
    mockBBox(visible, { x: 10, y: 10, width: 200, height: 100 });

    const offscreen = document.createElement('div');
    offscreen.setAttribute('data-testid', 'offscreen');
    document.body.appendChild(offscreen);
    mockBBox(offscreen, { x: 2000, y: 2000, width: 200, height: 100 });

    const result = captureViewportDOM();
    expect(result.length).toBe(1);
    expect(result[0].testId).toBe('visible');
  });

  it('captures aria attributes and roles', () => {
    const nav = document.createElement('nav');
    nav.setAttribute('role', 'navigation');
    nav.setAttribute('aria-label', 'Main menu');
    document.body.appendChild(nav);
    mockBBox(nav, { x: 0, y: 0, width: 1024, height: 60 });

    const result = captureViewportDOM();
    expect(result.length).toBe(1);
    expect(result[0].role).toBe('navigation');
    expect(result[0].ariaLabel).toBe('Main menu');
  });

  it('captures interactive element state', () => {
    const btn = document.createElement('button');
    btn.disabled = true;
    btn.textContent = 'Submit';
    document.body.appendChild(btn);
    mockBBox(btn, { x: 100, y: 100, width: 80, height: 30 });

    const result = captureViewportDOM();
    expect(result.length).toBe(1);
    expect(result[0].tag).toBe('button');
    expect(result[0].state?.enabled).toBe(false);
  });

  it('captures multiple visible children', () => {
    for (let i = 0; i < 3; i++) {
      const el = document.createElement('section');
      el.setAttribute('data-testid', `section-${i}`);
      document.body.appendChild(el);
      mockBBox(el, { x: 0, y: i * 200, width: 800, height: 180 });
    }

    const result = captureViewportDOM();
    expect(result.length).toBe(3);
    expect(result.map((n) => n.testId)).toEqual(['section-0', 'section-1', 'section-2']);
  });
});

// ---------------------------------------------------------------------------
// Viewport Size Variety
// ---------------------------------------------------------------------------

describe('viewport size variety', () => {
  const viewportSizes = [
    { w: 1920, h: 1080, name: 'Full HD' },
    { w: 3440, h: 1440, name: 'Ultra-wide' },
    { w: 768, h: 1024, name: 'Portrait tablet' },
    { w: 375, h: 667, name: 'Mobile' },
  ];

  beforeEach(() => {
    document.body.innerHTML = '';
  });

  for (const { w, h, name } of viewportSizes) {
    it(`captures elements correctly at ${name} (${w}x${h})`, () => {
      setViewportSize(w, h);

      const visible = document.createElement('div');
      visible.setAttribute('data-testid', 'visible');
      document.body.appendChild(visible);
      mockBBox(visible, { x: 10, y: 10, width: 200, height: 100 });

      const offscreen = document.createElement('div');
      offscreen.setAttribute('data-testid', 'offscreen');
      document.body.appendChild(offscreen);
      mockBBox(offscreen, { x: w + 500, y: h + 500, width: 200, height: 100 });

      const result = captureViewportDOM();
      expect(result.length).toBe(1);
      expect(result[0].testId).toBe('visible');
    });

    it(`correctly reports viewport intersection at ${name} (${w}x${h})`, () => {
      setViewportSize(w, h);

      const insideEl = document.createElement('div');
      mockBBox(insideEl, { x: 0, y: 0, width: w / 2, height: h / 2 });
      expect(isInViewport(insideEl)).toBe(true);

      const outsideEl = document.createElement('div');
      mockBBox(outsideEl, { x: w + 100, y: h + 100, width: 100, height: 100 });
      expect(isInViewport(outsideEl)).toBe(false);

      const partialEl = document.createElement('div');
      mockBBox(partialEl, { x: w - 50, y: h - 50, width: 100, height: 100 });
      expect(isInViewport(partialEl)).toBe(true);
    });
  }
});

// ---------------------------------------------------------------------------
// Constants verification
// ---------------------------------------------------------------------------

describe('constants', () => {
  it('MAX_TABLE_ROWS is 50', () => {
    expect(MAX_TABLE_ROWS).toBe(50);
  });

  it('MAX_TEXT_LENGTH is 500', () => {
    expect(MAX_TEXT_LENGTH).toBe(500);
  });

  it('MAX_TREE_DEPTH is 30', () => {
    expect(MAX_TREE_DEPTH).toBe(30);
  });

  it('MAX_NODE_COUNT is 3000', () => {
    expect(MAX_NODE_COUNT).toBe(3000);
  });
});
