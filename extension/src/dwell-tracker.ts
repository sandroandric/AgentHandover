/**
 * OpenMimic Observer — Dwell + Scroll Snapshot Triggers
 *
 * Extension-side dwell and scroll-reading detection that mirrors the
 * daemon's dwell tracker logic.
 *
 * Two categories of user input:
 *   - **Manipulation inputs** (click, keydown): reset the dwell timer
 *   - **Navigation inputs** (scroll, wheel): do NOT reset the dwell timer
 *
 * Timer logic:
 *   - On page load / focus: start dwell timer
 *   - On manipulation input: reset dwell timer, clear navigation tracking
 *   - On navigation input: record first navigation time since last manipulation
 *   - If timer reaches dwellThresholdMs without manipulation: fire onDwellSnapshot()
 *   - If navigation has been ongoing for scrollReadThresholdMs: fire onScrollSnapshot()
 *   - Each snapshot fires only once per dwell period (reset after manipulation)
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DwellConfig {
  /** Milliseconds of inactivity (no manipulation) before a dwell snapshot fires. Default: 3000 */
  dwellThresholdMs: number;
  /** Milliseconds of continuous scroll-reading before a scroll snapshot fires. Default: 8000 */
  scrollReadThresholdMs: number;
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

interface DwellState {
  /** Timestamp (ms) of the last manipulation input (click/keydown). */
  lastManipulationTime: number;
  /** Timestamp (ms) of the first navigation input since the last manipulation. null if no nav yet. */
  firstNavigationTime: number | null;
  /** Whether the dwell snapshot has already fired for this dwell period. */
  dwellFired: boolean;
  /** Whether the scroll-read snapshot has already fired for this period. */
  scrollFired: boolean;
  /** The interval ID for the periodic timer check. */
  timerInterval: ReturnType<typeof setInterval> | null;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Initialise the dwell and scroll-read tracker.
 *
 * @param config           Threshold configuration.
 * @param onDwellSnapshot  Called when the user has been dwelling (no manipulation)
 *                         for at least `dwellThresholdMs`.
 * @param onScrollSnapshot Called when the user has been scroll-reading for at
 *                         least `scrollReadThresholdMs`.
 * @returns                A cleanup function that removes all listeners and timers.
 */
export function initDwellTracker(
  config: DwellConfig,
  onDwellSnapshot: () => void,
  onScrollSnapshot: () => void,
): () => void {
  const state: DwellState = {
    lastManipulationTime: Date.now(),
    firstNavigationTime: null,
    dwellFired: false,
    scrollFired: false,
    timerInterval: null,
  };

  // -----------------------------------------------------------------------
  // Input handlers
  // -----------------------------------------------------------------------

  /**
   * Manipulation input: resets the dwell timer and clears navigation tracking.
   * Fires on click and keydown.
   */
  function handleManipulation(): void {
    state.lastManipulationTime = Date.now();
    state.firstNavigationTime = null;
    state.dwellFired = false;
    state.scrollFired = false;
  }

  /**
   * Navigation input: records the first navigation time since the last
   * manipulation. Does NOT reset the dwell timer.
   * Fires on scroll and wheel.
   */
  function handleNavigation(): void {
    if (state.firstNavigationTime === null) {
      state.firstNavigationTime = Date.now();
    }
  }

  /**
   * Focus handler: resets dwell timing when the page gains focus,
   * treating it as a fresh start for dwell detection.
   */
  function handleFocus(): void {
    state.lastManipulationTime = Date.now();
    state.firstNavigationTime = null;
    state.dwellFired = false;
    state.scrollFired = false;
  }

  // -----------------------------------------------------------------------
  // Timer check — runs periodically to evaluate thresholds
  // -----------------------------------------------------------------------

  function checkTimers(): void {
    const now = Date.now();

    // Dwell check: has enough time elapsed since the last manipulation?
    if (!state.dwellFired) {
      const elapsed = now - state.lastManipulationTime;
      if (elapsed >= config.dwellThresholdMs) {
        state.dwellFired = true;
        onDwellSnapshot();
      }
    }

    // Scroll-read check: has navigation been ongoing long enough?
    if (!state.scrollFired && state.firstNavigationTime !== null) {
      const scrollElapsed = now - state.firstNavigationTime;
      if (scrollElapsed >= config.scrollReadThresholdMs) {
        state.scrollFired = true;
        onScrollSnapshot();
      }
    }
  }

  // -----------------------------------------------------------------------
  // Set up listeners and timer
  // -----------------------------------------------------------------------

  // Manipulation inputs (capture phase for reliability)
  document.addEventListener('click', handleManipulation, true);
  document.addEventListener('keydown', handleManipulation, true);

  // Navigation inputs (capture phase)
  document.addEventListener('scroll', handleNavigation, true);
  document.addEventListener('wheel', handleNavigation, true);

  // Page focus
  window.addEventListener('focus', handleFocus);

  // Periodic timer check — 250ms gives good responsiveness without
  // excessive CPU usage
  state.timerInterval = setInterval(checkTimers, 250);

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------

  return () => {
    document.removeEventListener('click', handleManipulation, true);
    document.removeEventListener('keydown', handleManipulation, true);
    document.removeEventListener('scroll', handleNavigation, true);
    document.removeEventListener('wheel', handleNavigation, true);
    window.removeEventListener('focus', handleFocus);

    if (state.timerInterval !== null) {
      clearInterval(state.timerInterval);
      state.timerInterval = null;
    }
  };
}
