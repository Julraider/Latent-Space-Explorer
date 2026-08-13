/**
 * Einstiegspunkt des Viewers.
 *
 * Laedt einen Artefaktsatz, rendert ihn als instanzierte Quads und verbindet
 * Auswahl, Filter und Suche. Der Korpus ist echt (Met Open Access); ob die
 * Geometrie etwas bedeutet, sagt das Manifest — bei Platzhalter-Embeddings
 * steht die Warnung im Bild.
 */

import { Color, FogExp2, PerspectiveCamera, Scene, Vector2, Vector3, WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { loadArtifacts } from './artifacts.js';
import { createNeighborLines } from './neighborLines.js';
import { PICK_BUSY, Picker } from './picking.js';
import { createPromptMarker } from './promptMarker.js';
import {
  FLAG_MATCH,
  FLAG_NEIGHBOR,
  FLAG_SELECTED,
  FLAG_VISIBLE,
  createPointCloud,
} from './pointCloud.js';
import * as urlState from './urlState.js';

const params = new URLSearchParams(location.search);
const DATASET = params.get('data') ?? '/data/sample';

// `?webgl` erzwingt den WebGL2-Pfad. WebGPURenderer faellt normalerweise von
// selbst zurueck, wenn WebGPU fehlt — aber nicht, wenn WebGPU *vorhanden, aber
// kaputt* ist (aeltere Browser-Builds, experimentelle Treiber). Genau dann ist
// dieser Schalter der schnellste Weg, das Backend als Ursache auszuschliessen.
const FORCE_WEBGL = params.has('webgl');

// Nebel- und Hintergrundfarbe sind identisch: ferne Punkte loesen sich im
// Hintergrund auf, statt eine verwirrende Rauschwand dahinter zu bilden.
const BACKGROUND = new Color(0x07080c);

// Auf einem Retina-Panel ist das ein Performance-Faktor von ~1,8 fuer einen
// Unterschied, den bei weichen runden Sprites niemand sieht.
const MAX_PIXEL_RATIO = 1.5;

// Muss zu PALETTE in pipeline/src/lse/synthetic.py passen.
const PALETTE = [
  [230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66],
  [0, 114, 178], [213, 94, 0], [204, 121, 167], [148, 103, 189],
];

const FIELD_LABELS = {
  title: 'Titel',
  artist: 'Künstler',
  object_date: 'Datiert',
  medium: 'Material',
  classification: 'Gattung',
  object_number: 'Inventarnr.',
  link: 'Beim Met',
  department: 'Abteilung',
  culture: 'Kultur',
  period: 'Epoche',
};

const ui = {};
for (const id of [
  'status', 'status-detail', 'hud', 'help', 'legend', 'warning', 'controls',
  'search', 'search-count', 'filters', 'reset', 'detail', 'detail-title',
  'detail-fields', 'detail-neighbors', 'stat-points', 'stat-backend',
  'stat-frame', 'stat-draws', 'prompt', 'prompt-box', 'prompt-status',
]) {
  ui[id] = document.getElementById(id);
}

function fail(headline, detail) {
  ui.status.classList.remove('hidden');
  ui.status.classList.add('error');
  ui.status.querySelector('.headline').textContent = headline;
  ui['status-detail'].textContent = detail;
  console.error(headline, detail);
}

function renderLegend(labelNames) {
  if (!labelNames.length) return;
  ui.legend.replaceChildren(
    ...labelNames.map((name, index) => {
      const row = document.createElement('div');
      const swatch = document.createElement('i');
      swatch.style.background = `rgb(${PALETTE[index % PALETTE.length].join(',')})`;
      const text = document.createElement('span');
      text.textContent = name;
      row.append(swatch, text);
      return row;
    }),
  );
  ui.legend.hidden = false;
}

async function main() {
  ui['status-detail'].textContent = DATASET;

  let data;
  try {
    data = await loadArtifacts(DATASET);
  } catch (error) {
    fail('Artefakte konnten nicht geladen werden', String(error.message ?? error));
    return;
  }

  const scene = new Scene();
  scene.background = BACKGROUND;

  const radius = data.bounds.radius || 1;

  // Near/Far aus den echten Wolkengrenzen, nicht aus Vorgabewerten. Ein
  // Verhaeltnis um 1000 laesst einem 24-Bit-Tiefenpuffer reichlich Praezision;
  // das klassische `near = 0.1, far = 10000` ist die eigentliche Ursache von
  // Z-Fighting. Kein logarithmicDepthBuffer: der schreibt gl_FragDepth und
  // deaktiviert damit Early-Z, also genau den Mechanismus, der die Fuellrate
  // rettet.
  const camera = new PerspectiveCamera(55, 1, radius * 0.005, radius * 8);
  camera.position.set(radius * 1.6, radius * 0.9, radius * 1.6);

  // Nebeldichte aus dem Radius, gerechnet statt geraten: FogExp2 liefert
  // exp(-(d*k)^2), die Kamera steht bei ~2,4 Radien, die Wolke reicht also von
  // ~1,4 bis ~3,4 Radien. Mit k = 0,3/r bleibt die Vorderseite bei ~83 %
  // Sichtbarkeit und die Rueckseite faellt auf ~35 %.
  //
  // Wichtig: three wendet den Nebel auch bei gesetztem `fragmentNode` an
  // (NodeMaterial, setupOutput). Ein zu hoher Wert frisst die gesamte
  // Farbigkeit, ohne dass der Shader daran erkennbar schuld waere.
  scene.fog = new FogExp2(BACKGROUND.getHex(), 0.3 / radius);

  const renderer = new WebGPURenderer({ antialias: true, forceWebGL: FORCE_WEBGL });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(BACKGROUND);
  document.body.appendChild(renderer.domElement);

  try {
    await renderer.init();
  } catch (error) {
    fail('Renderer konnte nicht starten', String(error.message ?? error));
    return;
  }

  const cloud = createPointCloud(data, {
    maxPointPx: 48,
    background: [BACKGROUND.r, BACKGROUND.g, BACKGROUND.b],
  });
  scene.add(cloud);

  // Orbit um ein bewegliches Ziel, nicht Free-Fly: eine UMAP-Wolke ist ein
  // begrenztes Objekt ohne Boden, Horizont oder Massstab. Freies Fliegen darin
  // heisst ueberschiessen und im featurelosen Schwarz landen (docs/decisions.md E5).
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);
  controls.minDistance = radius * 0.05;
  controls.maxDistance = radius * 4;

  const picker = new Picker(renderer, scene, camera, cloud, data.n);

  const neighborLines = createNeighborLines(data.neighborsK || 20, radius);
  scene.add(neighborLines);

  const promptMarker = createPromptMarker({ worldSize: radius * 0.035 });
  scene.add(promptMarker);

  // ---------------------------------------------------------------------
  // Kamerafahrt
  // ---------------------------------------------------------------------

  let tween = null;

  /**
   * Faehrt Ziel und Kamera weich an eine neue Position.
   *
   * Nie springen: jede programmatische Bewegung ist ein Tween. Sprünge
   * desorientieren und werden als Fehler gelesen. Und weil das Ziel immer ein
   * echter Punkt ist, kann man sich dabei nicht verirren.
   */
  function flyTo(destination, distance, duration = 620) {
    const direction = camera.position.clone().sub(controls.target);
    if (direction.lengthSq() < 1e-9) direction.set(0, 0, 1);
    direction.setLength(distance);
    tween = {
      start: performance.now(),
      duration,
      fromTarget: controls.target.clone(),
      toTarget: destination.clone(),
      fromCamera: camera.position.clone(),
      toCamera: destination.clone().add(direction),
    };
  }

  function updateTween(now) {
    if (!tween) return;
    const t = Math.min(1, (now - tween.start) / tween.duration);
    // Ease-in-out; bei t=1 exakt am Ziel.
    const e = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
    controls.target.lerpVectors(tween.fromTarget, tween.toTarget, e);
    camera.position.lerpVectors(tween.fromCamera, tween.toCamera, e);
    if (t >= 1) tween = null;
  }

  const scratch = new Vector3();
  /** Weltposition eines Punktes — dieselbe Rechnung wie im Vertex-Shader. */
  function pointPosition(index, out = scratch) {
    const stride = data.strides.coord;
    const { offset, half } = data.dequant;
    return out.set(
      (data.coords[index * stride] / 32767) * half[0] + offset[0],
      (data.coords[index * stride + 1] / 32767) * half[1] + offset[1],
      (data.coords[index * stride + 2] / 32767) * half[2] + offset[2],
    );
  }

  function overview() {
    flyTo(new Vector3(0, 0, 0), radius * 2.4);
  }

  // ---------------------------------------------------------------------
  // Zustand: Auswahl, Filter, Suche
  // ---------------------------------------------------------------------

  let metadata = null;
  let neighbors = null;
  let selected = -1;
  const activeFilters = new Map(); // Facettenname -> Wertindex, -1 = alle

  const syncUrl = urlState.debounce(() =>
    urlState.write({
      camera: camera.position,
      target: controls.target,
      selected,
      query: ui.search.value,
      filters: Object.fromEntries(activeFilters),
    }),
  );

  function setFlag(index, bit, on) {
    if (on) cloud.flags[index] |= bit;
    else cloud.flags[index] &= ~bit;
  }

  function clearFlagEverywhere(bit) {
    for (let i = 0; i < cloud.flags.length; i++) cloud.flags[i] &= ~bit;
  }

  function applyFilters() {
    if (!metadata || activeFilters.size === 0) {
      for (let i = 0; i < cloud.flags.length; i++) cloud.flags[i] |= FLAG_VISIBLE;
      cloud.commitFlags();
      return;
    }
    const columns = [...activeFilters.entries()]
      .filter(([, value]) => value >= 0)
      .map(([field, value]) => [metadata.facetFields.indexOf(field), value]);

    for (let i = 0; i < data.n; i++) {
      let visible = true;
      for (const [column, value] of columns) {
        if (metadata.facetIndex(i, column) !== value) {
          visible = false;
          break;
        }
      }
      setFlag(i, FLAG_VISIBLE, visible);
    }
    cloud.commitFlags();
    syncUrl();
  }

  function runSearch(query) {
    clearFlagEverywhere(FLAG_MATCH);
    if (!metadata || !query.trim()) {
      cloud.uniforms.searchActive.value = 0;
      ui['search-count'].textContent = '';
      cloud.commitFlags();
      syncUrl();
      return;
    }
    const hits = metadata.search(query.trim());
    for (const index of hits) cloud.flags[index] |= FLAG_MATCH;
    cloud.uniforms.searchActive.value = 1;
    ui['search-count'].textContent =
      hits.length === 0
        ? 'keine Treffer'
        : `${hits.length.toLocaleString('de-DE')} Treffer`;
    cloud.commitFlags();
    syncUrl();

    // Zum ersten Treffer fliegen: sonst "findet" die Suche Punkte tief in der
    // Wolke, und man sieht drei davon.
    if (hits.length) flyTo(pointPosition(hits[0], new Vector3()), radius * 0.6);
  }

  async function select(index) {
    if (selected >= 0) {
      setFlag(selected, FLAG_SELECTED, false);
      clearFlagEverywhere(FLAG_NEIGHBOR);
    }
    selected = index;

    if (index < 0) {
      cloud.commitFlags();
      neighborLines.clear();
      ui.detail.hidden = true;
      syncUrl();
      return;
    }
    setFlag(index, FLAG_SELECTED, true);

    // Die echten hochdimensionalen Nachbarn hervorheben — nicht die in 3D.
    // Genau die Abweichung zwischen beidem ist die Aussage: wo ein Nachbar auf
    // der gegenueberliegenden Seite landet, sieht man die Projektionsverzerrung.
    neighbors ??= await data.loadNeighbors();
    let farthest = 0;
    if (neighbors) {
      const k = data.neighborsK;
      const origin = pointPosition(index, new Vector3());
      const targets = [];
      for (let j = 0; j < k; j++) {
        const neighbor = neighbors[index * k + j];
        if (neighbor >= data.n) continue;
        setFlag(neighbor, FLAG_NEIGHBOR, true);
        const position = pointPosition(neighbor, new Vector3());
        farthest = Math.max(farthest, origin.distanceTo(position));
        targets.push(position.toArray());
      }
      neighborLines.setNeighbors(origin.toArray(), targets);
    }
    cloud.commitFlags();
    showDetail(index, farthest);
    syncUrl();
  }

  function showDetail(index, farthestNeighbor = 0) {
    // Der Index steht im DOM: er ist die ID des Punktes, und ohne ihn ist jeder
    // Fehlerbericht ueber "das falsche Objekt" nicht nachvollziehbar.
    ui.detail.dataset.index = String(index);
    const record = metadata?.record(index);
    ui['detail-title'].textContent = record?.title || `Punkt ${index}`;

    const rows = [];
    // `seen` verhindert Doppelzeilen: Felder wie `classification` sind sowohl
    // Facette als auch Textfeld und wuerden sonst zweimal erscheinen.
    const seen = new Set(['title']);

    const labelName = data.labelNames[data.labels[index]];
    if (labelName) {
      rows.push([FIELD_LABELS.department, labelName]);
      seen.add('department');
    }
    if (metadata) {
      for (const field of metadata.facetFields) {
        if (seen.has(field)) continue;
        const value = metadata.facet(index, field);
        if (value && value !== 'unbekannt') rows.push([FIELD_LABELS[field] ?? field, value]);
        seen.add(field);
      }
      for (const field of metadata.fields) {
        if (seen.has(field)) continue;
        const value = record?.[field];
        if (value) rows.push([FIELD_LABELS[field] ?? field, value]);
        seen.add(field);
      }
    }

    ui['detail-fields'].replaceChildren(
      ...rows.flatMap(([name, value]) => {
        const dt = document.createElement('dt');
        dt.textContent = name;
        const dd = document.createElement('dd');
        if (/^https?:\/\//.test(value)) {
          const link = document.createElement('a');
          link.href = value;
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
          link.textContent = 'Sammlungsseite öffnen';
          dd.append(link);
        } else {
          dd.textContent = value;
        }
        return [dt, dd];
      }),
    );

    if (neighbors) {
      // Der entfernteste echte Nachbar, gemessen am Wolkenradius. Genau diese
      // Zahl ist die Aussage: liegt sie hoch, hat die Projektion Punkte
      // auseinandergerissen, die im Modell benachbart sind.
      const spread = farthestNeighbor / radius;
      const verdict =
        spread > 0.6
          ? 'Ein Teil davon liegt quer im Raum — dort hat die Projektion Nachbarschaft verloren.'
          : 'Sie liegen alle nah beieinander — hier bildet die Projektion gut ab.';
      ui['detail-neighbors'].textContent =
        `${data.neighborsK} nächste Nachbarn im hochdimensionalen Raum, verbunden. `
        + `Der entfernteste liegt bei ${(spread * 100).toFixed(0)} % des Wolkenradius. `
        + verdict;
    } else {
      ui['detail-neighbors'].textContent = '';
    }
    ui.detail.hidden = false;
  }

  function buildFilterUI() {
    if (!metadata) return;
    ui.filters.replaceChildren(
      ...metadata.facetFields.map((field) => {
        const label = document.createElement('label');
        label.textContent = FIELD_LABELS[field] ?? field;
        const dropdown = document.createElement('select');
        const all = document.createElement('option');
        all.value = '-1';
        all.textContent = 'alle';
        dropdown.append(all);
        metadata.facetValues[field].forEach((value, index) => {
          const option = document.createElement('option');
          option.value = String(index);
          option.textContent = value || '(leer)';
          dropdown.append(option);
        });
        dropdown.addEventListener('change', () => {
          activeFilters.set(field, Number(dropdown.value));
          applyFilters();
        });
        label.append(dropdown);
        return label;
      }),
    );
  }

  // ---------------------------------------------------------------------
  // Eingaben
  // ---------------------------------------------------------------------

  let downAt = null;
  renderer.domElement.addEventListener('pointerdown', (event) => {
    downAt = { x: event.clientX, y: event.clientY };
  });

  renderer.domElement.addEventListener('pointerup', async (event) => {
    // Nur als Klick werten, wenn kaum bewegt wurde — sonst selektiert jedes
    // Drehen der Kamera einen Punkt.
    if (!downAt) return;
    const moved = Math.hypot(event.clientX - downAt.x, event.clientY - downAt.y);
    downAt = null;
    if (moved > 4 || event.button !== 0) return;

    const rect = renderer.domElement.getBoundingClientRect();
    const index = await picker.pick(event.clientX - rect.left, event.clientY - rect.top);
    // Verworfene Anfragen ignorieren, nicht als "nichts getroffen" behandeln:
    // ein Doppelklick loest drei ueberlappende Anfragen aus.
    if (index === PICK_BUSY) return;
    await select(index);
  });

  renderer.domElement.addEventListener('dblclick', async (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    const index = await picker.pick(event.clientX - rect.left, event.clientY - rect.top);
    if (index === PICK_BUSY) return;
    if (index < 0) {
      overview();
      return;
    }
    await select(index);
    flyTo(pointPosition(index, new Vector3()), radius * 0.35);
  });

  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      select(-1);
      overview();
    }
  });

  ui.detail.querySelector('.close').addEventListener('click', () => select(-1));

  let searchTimer = 0;
  ui.search.addEventListener('input', () => {
    clearTimeout(searchTimer);
    // Entprellt: bei jedem Tastendruck ueber alle Titel zu suchen und die Flags
    // hochzuladen waere Arbeit, die der Nutzer nie sieht.
    searchTimer = setTimeout(() => runSearch(ui.search.value), 150);
  });

  ui.reset.addEventListener('click', () => {
    ui.search.value = '';
    runSearch('');
    activeFilters.clear();
    promptMarker.hide();
    ui['prompt-status'].textContent = '';
    for (const dropdown of ui.filters.querySelectorAll('select')) dropdown.value = '-1';
    applyFilters();
    select(-1);
    overview();
  });

  // ---------------------------------------------------------------------

  ui['stat-points'].textContent = data.n.toLocaleString('de-DE');
  ui['stat-backend'].textContent = renderer.backend?.isWebGPUBackend
    ? 'WebGPU'
    : 'WebGL2 (Fallback)';
  renderLegend(data.labelNames);
  ui.hud.hidden = false;
  ui.help.hidden = false;

  // Platzhalter-Geometrie muss im Bild sichtbar sein, nicht nur im Manifest.
  // Ein Raum, der aussieht wie ein Ergebnis, aber keines ist, waere sonst genau
  // die Illusion, gegen die das Go/No-Go-Tor gebaut wurde.
  if (data.manifest.projection?.placeholder_embeddings) {
    ui.warning.textContent =
      'Platzhalter-Embeddings: die Metadaten sind echt, die Anordnung im Raum '
      + 'bedeutet nichts. Erst `lse embed` liefert eine echte Geometrie.';
    ui.warning.hidden = false;
  }

  ui.status.classList.add('hidden');

  // Metadaten nachladen, ohne das erste Bild aufzuhalten.
  const restored = urlState.read();
  data.loadMetadata().then((loaded) => {
    if (!loaded) return;
    metadata = loaded;
    buildFilterUI();
    ui.controls.hidden = false;

    // Zustand aus der URL erst hier anwenden: Filter und Suche brauchen die
    // Metadaten, und eine halb angewandte Ansicht waere schlimmer als keine.
    if (!restored) return;
    if (restored.filters) {
      for (const [field, value] of Object.entries(restored.filters)) {
        activeFilters.set(field, value);
        const dropdown = [...ui.filters.querySelectorAll('select')][
          metadata.facetFields.indexOf(field)
        ];
        if (dropdown) dropdown.value = String(value);
      }
      applyFilters();
    }
    if (restored.query) {
      ui.search.value = restored.query;
      runSearch(restored.query);
    }
    if (restored.selected >= 0 && restored.selected < data.n) {
      select(restored.selected);
    }
  });

  // Kamera dagegen sofort, damit kein sichtbarer Sprung entsteht.
  if (restored?.camera && restored?.target) {
    camera.position.fromArray(restored.camera);
    controls.target.fromArray(restored.target);
    controls.update();
  }
  controls.addEventListener('change', syncUrl);

  // ---------------------------------------------------------------------
  // Live-Prompt (optionaler lokaler Dienst)
  // ---------------------------------------------------------------------

  // Das Prompt-Feld erscheint nur, wenn ein Dienst antwortet. Ohne ihn bleibt
  // alles andere voll funktionsfaehig — der Viewer ist eine statische Seite,
  // und ein Feature, das einen laufenden Python-Prozess und eine dicke GPU
  // voraussetzt, darf sie nicht als Ganzes unbenutzbar machen.
  const SERVICE = params.get('service') ?? 'http://127.0.0.1:8765';
  let serviceInfo = null;

  fetch(`${SERVICE}/health`, { signal: AbortSignal.timeout(2500) })
    .then((response) => (response.ok ? response.json() : null))
    .then((info) => {
      if (!info || info.n !== data.n) return;
      serviceInfo = info;
      ui['prompt-box'].hidden = false;
      if (info.stub) {
        ui['prompt-status'].className = 'stub';
        ui['prompt-status'].textContent =
          'Dienst im Stub-Modus: die Platzierung ist reproduzierbar, aber ohne Bedeutung.';
      }
    })
    .catch(() => {
      /* Kein Dienst — das Feld bleibt verborgen. Kein Fehler. */
    });

  async function submitPrompt(text) {
    if (!serviceInfo || !text.trim()) return;
    ui['prompt-status'].className = '';
    ui['prompt-status'].textContent = 'platziere …';
    try {
      const response = await fetch(`${SERVICE}/prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
        signal: AbortSignal.timeout(30000),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error ?? `HTTP ${response.status}`);

      promptMarker.setPosition(result.position, { warn: !result.confident });

      // Die Nachbarn, aus denen die Position gemittelt wurde, sichtbar machen:
      // ohne sie ist der Marker eine Behauptung.
      clearFlagEverywhere(FLAG_NEIGHBOR);
      const targets = [];
      for (const neighbor of result.neighbors) {
        if (neighbor >= data.n) continue;
        setFlag(neighbor, FLAG_NEIGHBOR, true);
        targets.push(pointPosition(neighbor, new Vector3()).toArray());
      }
      cloud.commitFlags();
      neighborLines.setNeighbors(result.position, targets);

      flyTo(new Vector3().fromArray(result.position), radius * 0.8);

      const spread = `${(result.spread * 100).toFixed(0)} % Streuung`;
      if (result.stub) {
        ui['prompt-status'].className = 'stub';
        ui['prompt-status'].textContent =
          `Stub-Modus — ${result.neighbors.length} Nachbarn, ${spread}. Ohne Bedeutung.`;
      } else if (!result.confident) {
        ui['prompt-status'].className = 'warn';
        ui['prompt-status'].textContent = result.note;
      } else {
        ui['prompt-status'].className = '';
        ui['prompt-status'].textContent =
          `Platziert zwischen ${result.neighbors.length} nächsten Nachbarn (${spread}).`;
      }
    } catch (error) {
      ui['prompt-status'].className = 'warn';
      ui['prompt-status'].textContent = `Dienst antwortet nicht: ${error.message ?? error}`;
    }
  }

  ui.prompt.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') submitPrompt(ui.prompt.value);
  });

  function resize() {
    const { innerWidth: w, innerHeight: h } = window;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
    renderer.setSize(w, h);
  }
  window.addEventListener('resize', resize);
  resize();

  // WebGL-Kontextverlust ist real: er tritt bei GPU-Umschaltung im Laptop und
  // beim Backgrounding von Tabs auf. Unbehandelt sieht der Nutzer ein dauerhaft
  // eingefrorenes Bild ohne jede Erklaerung.
  renderer.domElement.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    fail('Grafikkontext verloren', 'Seite neu laden.');
  });

  let frameAccum = 0;
  let frameCount = 0;
  let lastReport = performance.now();
  const drawingBuffer = new Vector2();

  renderer.setAnimationLoop(() => {
    const started = performance.now();

    updateTween(started);
    controls.update();
    // Der Pixel-Clamp haengt an Bildhoehe und Sichtfeld — beides kann sich
    // jederzeit aendern.
    const bufferHeight = renderer.getDrawingBufferSize(drawingBuffer).y;
    cloud.updateScreenMetrics(camera, bufferHeight);
    promptMarker.updateScreenMetrics(camera, bufferHeight);
    renderer.render(scene, camera);

    frameAccum += performance.now() - started;
    frameCount += 1;
    if (started - lastReport > 500) {
      ui['stat-frame'].textContent = `${(frameAccum / frameCount).toFixed(1)} ms`;
      ui['stat-draws'].textContent = String(renderer.info?.render?.drawCalls ?? '—');
      frameAccum = 0;
      frameCount = 0;
      lastReport = started;
    }
  });
}

main();
