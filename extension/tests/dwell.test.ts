/**
 * Tests for Dwell + Scroll Snapshot Triggers module.
 *
 * Verifies:
 *   - Dwell timer fires after threshold without manipulation
 *   - Manipulation inputs (click, keydown) reset the dwell timer
 *   - Navigation inputs (scroll, wheel) do NOT reset the dwell timer
 *   - Scroll-read fires after scrollReadThresholdMs of continuous navigation
 *   - Each snapshot fires only once per dwell period
 *   - Cleanup removes all listeners and timers
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { initDwellTracker, type DwellConfig } from '../src/dwell-tracker';

// ---------------------------------------------------------------------------
// Mock chrome.runtime.sendMessage
// ---------------------------------------------------------------------------

beforeEach(() => {
  (globalThis as Record<string, unknown>).chrome = {
    runtime: {
      sendMessage: vi.fn(),
      lastError: null,
    },
  };
  // Use fake timers for precise control
  vi.useFakeTimers();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
  delete (globalThis as Record<string, unknown>).chrome;
});

// ---------------------------------------------------------------------------
// Helper to create a config with short thresholds for testing
// ---------------------------------------------------------------------------

function testConfig(overrides: Partial<DwellConfig> = {}): DwellConfig {
  return {
    dwellThresholdMs: 1000,
    scrollReadThresholdMs: 3000,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Dwell timer tests
// ---------------------------------------------------------------------------

describe('Dwell timer', () => {
  it('should fire onDwellSnapshot after dwellThresholdMs without manipulation', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Advance past the dwell threshold
    vi.advanceTimersByTime(1250); // 1000ms threshold + 250ms timer interval

    expect(onDwell).toHaveBeenCalledOnce();
    expect(onScroll).not.toHaveBeenCalled();

    cleanup();
  });

  it('should not fire dwell snapshot before threshold', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Advance to just before the threshold
    vi.advanceTimersByTime(750);

    expect(onDwell).not.toHaveBeenCalled();

    cleanup();
  });

  it('should fire dwell snapshot only once per period', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Advance well past the threshold
    vi.advanceTimersByTime(5000);

    expect(onDwell).toHaveBeenCalledOnce();

    cleanup();
  });

  it('should reset dwell timer on click (manipulation)', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Advance partway through
    vi.advanceTimersByTime(750);

    // Simulate a click (manipulation input)
    document.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    // Advance another 750ms (would have been 1500ms total, but timer reset)
    vi.advanceTimersByTime(750);

    // Should NOT have fired because the click reset the timer
    expect(onDwell).not.toHaveBeenCalled();

    // Now advance past the full threshold from the click
    vi.advanceTimersByTime(500);

    expect(onDwell).toHaveBeenCalledOnce();

    cleanup();
  });

  it('should reset dwell timer on keydown (manipulation)', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Advance partway
    vi.advanceTimersByTime(750);

    // Simulate keydown
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'a', bubbles: true }));

    // Advance 750ms more
    vi.advanceTimersByTime(750);

    // Should not have fired yet
    expect(onDwell).not.toHaveBeenCalled();

    cleanup();
  });

  it('should NOT reset dwell timer on scroll (navigation)', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Advance partway
    vi.advanceTimersByTime(750);

    // Simulate scroll (navigation input — should NOT reset dwell timer)
    document.dispatchEvent(new Event('scroll'));

    // Advance past original threshold
    vi.advanceTimersByTime(500);

    // Dwell should fire because scroll does not reset the dwell timer
    expect(onDwell).toHaveBeenCalledOnce();

    cleanup();
  });

  it('should NOT reset dwell timer on wheel (navigation)', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Advance partway
    vi.advanceTimersByTime(750);

    // Simulate wheel event (navigation)
    document.dispatchEvent(new WheelEvent('wheel'));

    // Advance past original threshold
    vi.advanceTimersByTime(500);

    // Dwell should still fire
    expect(onDwell).toHaveBeenCalledOnce();

    cleanup();
  });

  it('should allow dwell to fire again after manipulation resets it', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // First dwell
    vi.advanceTimersByTime(1250);
    expect(onDwell).toHaveBeenCalledOnce();

    // Manipulation resets the period
    document.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    // Second dwell
    vi.advanceTimersByTime(1250);
    expect(onDwell).toHaveBeenCalledTimes(2);

    cleanup();
  });
});

// ---------------------------------------------------------------------------
// Scroll-read timer tests
// ---------------------------------------------------------------------------

describe('Scroll-read timer', () => {
  it('should fire onScrollSnapshot after scrollReadThresholdMs of continuous navigation', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Start scrolling
    document.dispatchEvent(new Event('scroll'));

    // Advance past scroll-read threshold
    vi.advanceTimersByTime(3250); // 3000ms + 250ms interval

    expect(onScroll).toHaveBeenCalledOnce();

    cleanup();
  });

  it('should not fire scroll snapshot before threshold', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Start scrolling
    document.dispatchEvent(new Event('scroll'));

    // Advance to just before threshold
    vi.advanceTimersByTime(2500);

    expect(onScroll).not.toHaveBeenCalled();

    cleanup();
  });

  it('should fire scroll snapshot only once per period', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Start scrolling
    document.dispatchEvent(new Event('scroll'));

    // Advance well past threshold
    vi.advanceTimersByTime(10000);

    expect(onScroll).toHaveBeenCalledOnce();

    cleanup();
  });

  it('should reset scroll tracking on manipulation (click)', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Start scrolling
    document.dispatchEvent(new Event('scroll'));

    // Advance partway through scroll-read threshold
    vi.advanceTimersByTime(2000);

    // Manipulation resets everything including navigation tracking
    document.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    // More time passes but no new scroll
    vi.advanceTimersByTime(2000);

    // Scroll snapshot should NOT have fired (navigation tracking was reset)
    expect(onScroll).not.toHaveBeenCalled();

    cleanup();
  });

  it('should track first navigation time correctly after manipulation', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Manipulation
    document.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    // Start scrolling after manipulation
    vi.advanceTimersByTime(500);
    document.dispatchEvent(new Event('scroll'));

    // Advance past scroll-read threshold from the first scroll event
    vi.advanceTimersByTime(3250);

    expect(onScroll).toHaveBeenCalledOnce();

    cleanup();
  });

  it('should fire both dwell and scroll snapshots independently', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const config = testConfig({ dwellThresholdMs: 1000, scrollReadThresholdMs: 2000 });
    const cleanup = initDwellTracker(config, onDwell, onScroll);

    // Start scrolling immediately
    document.dispatchEvent(new Event('scroll'));

    // After 1250ms: dwell should fire (no manipulation)
    vi.advanceTimersByTime(1250);
    expect(onDwell).toHaveBeenCalledOnce();
    expect(onScroll).not.toHaveBeenCalled();

    // After another 1000ms (total 2250ms): scroll should fire
    vi.advanceTimersByTime(1000);
    expect(onScroll).toHaveBeenCalledOnce();

    cleanup();
  });
});

// ---------------------------------------------------------------------------
// Focus handling tests
// ---------------------------------------------------------------------------

describe('Focus handling', () => {
  it('should reset dwell timer on window focus', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    // Advance partway
    vi.advanceTimersByTime(750);

    // Window gains focus
    window.dispatchEvent(new FocusEvent('focus'));

    // Advance same amount — should not have fired because focus reset the timer
    vi.advanceTimersByTime(750);

    expect(onDwell).not.toHaveBeenCalled();

    // Now advance past threshold from the focus event
    vi.advanceTimersByTime(500);

    expect(onDwell).toHaveBeenCalledOnce();

    cleanup();
  });
});

// ---------------------------------------------------------------------------
// Cleanup tests
// ---------------------------------------------------------------------------

describe('Cleanup', () => {
  it('should stop timers and listeners after cleanup', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    cleanup();

    // Advance well past all thresholds
    vi.advanceTimersByTime(10000);

    expect(onDwell).not.toHaveBeenCalled();
    expect(onScroll).not.toHaveBeenCalled();
  });

  it('should not respond to events after cleanup', () => {
    const onDwell = vi.fn();
    const onScroll = vi.fn();
    const cleanup = initDwellTracker(testConfig(), onDwell, onScroll);

    cleanup();

    // These events should be ignored
    document.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    document.dispatchEvent(new Event('scroll'));

    vi.advanceTimersByTime(10000);

    expect(onDwell).not.toHaveBeenCalled();
    expect(onScroll).not.toHaveBeenCalled();
  });
});
