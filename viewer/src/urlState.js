/**
 * Ansichtszustand im Location-Hash.
 *
 * Fuer ein Portfolio-Stueck ist das der Unterschied zwischen "schau dir das mal
 * an" und "schau dir **das hier** an": ein Link fuehrt direkt auf die
 * interessante Stelle, mit Kamera, Filter und Auswahl. Kostet ein paar Zeilen.
 *
 * Zahlen werden gerundet gespeichert. Ein Hash mit siebzehn Nachkommastellen
 * ist nicht kopierbar, und die Praezision traegt nichts bei: die Kamera steht
 * ohnehin nie exakt wieder gleich.
 */

const PRECISION = 2;

function round(value) {
  return Number(value.toFixed(PRECISION));
}

/** Liest den Zustand aus dem Hash. Unlesbares wird ignoriert, nicht geworfen. */
export function read() {
  const raw = location.hash.replace(/^#/, '');
  if (!raw) return null;
  try {
    const params = new URLSearchParams(raw);
    const state = {};

    const camera = params.get('c');
    if (camera) {
      const parts = camera.split(',').map(Number);
      if (parts.length === 6 && parts.every(Number.isFinite)) {
        state.camera = parts.slice(0, 3);
        state.target = parts.slice(3, 6);
      }
    }

    const selected = params.get('s');
    if (selected !== null && Number.isFinite(Number(selected))) {
      state.selected = Number(selected);
    }

    const query = params.get('q');
    if (query) state.query = query;

    const filters = params.get('f');
    if (filters) {
      state.filters = {};
      for (const pair of filters.split(',')) {
        const [field, value] = pair.split(':');
        if (field && Number.isFinite(Number(value))) state.filters[field] = Number(value);
      }
    }
    return state;
  } catch {
    return null;
  }
}

/**
 * Schreibt den Zustand in den Hash.
 *
 * `replaceState` statt Zuweisung an `location.hash`: sonst waechst der
 * Browserverlauf bei jeder Kamerabewegung um einen Eintrag, und der
 * Zurueck-Knopf wird unbenutzbar.
 */
export function write({ camera, target, selected, query, filters }) {
  const params = new URLSearchParams();

  if (camera && target) {
    params.set(
      'c',
      [...camera.toArray(), ...target.toArray()].map(round).join(','),
    );
  }
  if (selected >= 0) params.set('s', String(selected));
  if (query) params.set('q', query);

  const active = Object.entries(filters ?? {}).filter(([, value]) => value >= 0);
  if (active.length) {
    params.set('f', active.map(([field, value]) => `${field}:${value}`).join(','));
  }

  const next = `#${params.toString()}`;
  if (next !== location.hash) {
    history.replaceState(null, '', next);
  }
}

/**
 * Ruft `callback` erst auf, wenn sich `interval` Millisekunden nichts mehr
 * getan hat.
 *
 * Die Kamera bewegt sich in jedem Frame; den Hash 60-mal pro Sekunde zu
 * schreiben belastet den Hauptthread ohne jeden Gegenwert.
 */
export function debounce(callback, interval = 400) {
  let timer = 0;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => callback(...args), interval);
  };
}
