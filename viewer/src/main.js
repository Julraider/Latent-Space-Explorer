/**
 * Einstiegspunkt des Viewers.
 *
 * Phase 0a: Gruest. Laedt einen Artefaktsatz, rendert ihn als instanzierte
 * Quads und beweist damit den Datenvertrag Ende zu Ende. Der Korpus ist noch
 * synthetisch — echte Met-Daten kommen in Phase 0b.
 */

import { Color, FogExp2, PerspectiveCamera, Scene, Vector2, WebGPURenderer } from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { loadArtifacts } from './artifacts.js';
import { createPointCloud } from './pointCloud.js';

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

// Muss zu PALETTE in pipeline/src/lse/synthetic.py passen. Die Farben stehen
// bereits pro Punkt in colors.u8.bin — die Legende braucht sie zusaetzlich, um
// Labelnamen zuzuordnen, ohne die Punktdaten zurueckzulesen.
const PALETTE = [
  [230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66],
  [0, 114, 178], [213, 94, 0], [204, 121, 167], [148, 103, 189],
];

const ui = {
  status: document.getElementById('status'),
  statusDetail: document.getElementById('status-detail'),
  hud: document.getElementById('hud'),
  help: document.getElementById('help'),
  legend: document.getElementById('legend'),
  warning: document.getElementById('warning'),
  points: document.getElementById('stat-points'),
  backend: document.getElementById('stat-backend'),
  frame: document.getElementById('stat-frame'),
  draws: document.getElementById('stat-draws'),
};

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

function fail(headline, detail) {
  ui.status.classList.remove('hidden');
  ui.status.classList.add('error');
  ui.status.querySelector('.headline').textContent = headline;
  ui.statusDetail.textContent = detail;
  console.error(headline, detail);
}

async function main() {
  ui.statusDetail.textContent = DATASET;

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

  // Nebeldichte aus dem Radius: ein UMAP-Neulauf aendert den Massstab, und eine
  // handgetunte Konstante waere danach still falsch.
  //
  // Der Faktor ist gerechnet, nicht geraten. FogExp2 liefert exp(-(d*k)^2); die
  // Kamera steht bei ~2,4 Radien, die Wolke reicht also von ~1,4 bis ~3,4
  // Radien. Mit k = 0,3/r bleibt die Vorderseite bei ~83 % Sichtbarkeit und die
  // Rueckseite faellt auf ~35 % — ein lesbarer Tiefenhinweis, statt die Wolke
  // flaechendeckend abzudunkeln.
  //
  // Wichtig: three wendet den Nebel auch bei gesetztem `fragmentNode` an
  // (NodeMaterial macht das in setupOutput, nach dem eigenen Knoten). Ein zu
  // hoher Wert frisst also die gesamte Farbigkeit, ohne dass der Shader daran
  // erkennbar schuld waere.
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

  const cloud = createPointCloud(data, { maxPointPx: 48 });
  scene.add(cloud);

  // Orbit um ein bewegliches Ziel, nicht Free-Fly: eine UMAP-Wolke ist ein
  // begrenztes Objekt ohne Boden, Horizont oder Massstab. Freies Fliegen darin
  // heisst ueberschiessen und im featurelosen Schwarz landen (docs/decisions.md E5).
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);
  controls.minDistance = radius * 0.05;
  // Welt-Up ist gesperrt und es gibt keinen Roll — der groesste Ausloeser fuer
  // Motion Sickness im Browser-3D. OrbitControls macht das per Vorgabe richtig,
  // TrackballControls nicht.
  controls.maxDistance = radius * 4;

  ui.points.textContent = data.n.toLocaleString('de-DE');
  ui.backend.textContent = renderer.backend?.isWebGPUBackend ? 'WebGPU' : 'WebGL2 (Fallback)';
  renderLegend(data.labelNames);
  ui.hud.hidden = false;
  ui.help.hidden = false;

  // Platzhalter-Geometrie muss im Bild sichtbar sein, nicht nur im Manifest.
  // Ein Raum, der aussieht wie ein Ergebnis, aber keines ist, waere sonst genau
  // die Illusion, gegen die das Go/No-Go-Tor gebaut wurde.
  if (data.manifest.projection?.placeholder_embeddings) {
    ui.warning.textContent =
      'Platzhalter-Embeddings: die Metadaten sind echt, die Anordnung im Raum ' +
      'bedeutet nichts. Erst `lse embed` liefert eine echte Geometrie.';
    ui.warning.hidden = false;
  }

  ui.status.classList.add('hidden');

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
  // Einmal allokiert: pro Frame ein neues Objekt zu erzeugen laedt den GC
  // ohne Gegenwert ein.
  const drawingBuffer = new Vector2();

  renderer.setAnimationLoop(() => {
    const started = performance.now();

    controls.update();
    // Der Pixel-Clamp haengt an Bildhoehe und Sichtfeld — beides kann sich
    // jederzeit aendern.
    cloud.updateScreenMetrics(camera, renderer.getDrawingBufferSize(drawingBuffer).y);
    renderer.render(scene, camera);

    frameAccum += performance.now() - started;
    frameCount += 1;
    if (started - lastReport > 500) {
      ui.frame.textContent = `${(frameAccum / frameCount).toFixed(1)} ms`;
      ui.draws.textContent = String(renderer.info?.render?.drawCalls ?? '—');
      frameAccum = 0;
      frameCount = 0;
      lastReport = started;
    }
  });
}

main();
