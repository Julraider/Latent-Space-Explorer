/**
 * Die Punktwolke: eine Geometrie, instanzierte Quads.
 *
 * Bewusst NICHT `THREE.Points` / `gl.POINTS`, aus vier unabhaengigen Gruenden
 * (siehe docs/decisions.md E3):
 *
 *  1. `gl_PointSize` ist treiberseitig gedeckelt — auf Apple Silicon bei 64 px.
 *  2. `gl.POINTS` clippt am Mittelpunkt, nicht an der Ausdehnung; grosse Sprites
 *     verschwinden schlagartig am Bildrand.
 *  3. WebGPU hat ueberhaupt keine Punktgroesse: `point-list` rendert 1x1 Pixel.
 *  4. Rundung, Rand, Groessen-Clamp und Filterzustaende brauchen Kontrolle pro
 *     Fragment.
 *
 * Der Vertex-Knoten wird von Anzeige- und Picking-Material **geteilt**. Das ist
 * der eigentliche Grund fuer GPU-ID-Picking: die Trefferflaeche stimmt damit
 * automatisch mit Groessenabschwaechung, Groessen-Clamp, runder Maske,
 * Tiefenverdeckung und jedem Filterzustand ueberein, ohne dass irgendwo eine
 * zweite Kopie dieser Logik gepflegt werden muss.
 */

import {
  Box3,
  BufferAttribute,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  MeshBasicNodeMaterial,
  Sphere,
  Vector3,
} from 'three/webgpu';
import {
  Discard,
  Fn,
  attribute,
  cameraProjectionMatrix,
  float,
  floor,
  instanceIndex,
  mix,
  mod,
  modelViewMatrix,
  smoothstep,
  uniform,
  varying,
  vec3,
  vec4,
} from 'three/tsl';

/** Bits im `aFlags`-Attribut. Muss zu interaction.js passen. */
export const FLAG_VISIBLE = 1; // erfuellt den aktiven Facettenfilter
export const FLAG_MATCH = 2; // Treffer der Stichwortsuche
export const FLAG_SELECTED = 4; // angeklickt
export const FLAG_NEIGHBOR = 8; // echter hochdimensionaler Nachbar des Selektierten

export function createPointCloud(data, options = {}) {
  const { maxPointPx = 48, sizeFactor = 0.25, background = [0.027, 0.031, 0.047] } = options;
  const { n, coords, colors, dequant, bounds, strides } = data;

  const geometry = new InstancedBufferGeometry();

  // Template-Quad in lokalen Koordinaten [-0.5, 0.5]. Vier Vertices, zwei
  // Dreiecke — von allen Instanzen geteilt.
  geometry.setAttribute(
    'position',
    new BufferAttribute(
      new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]),
      3,
    ),
  );
  geometry.setIndex([0, 1, 2, 0, 2, 3]);

  // Die eigentlichen Daten, ohne Umkopieren direkt aus der Binaerdatei.
  // `normalized: true` heisst: die GPU liefert [-1, 1] bzw. [0, 1], die
  // Rueckrechnung auf Weltkoordinaten passiert im Shader.
  geometry.setAttribute('aPos', new InstancedBufferAttribute(coords, strides.coord, true));
  geometry.setAttribute('aColor', new InstancedBufferAttribute(colors, strides.color, true));

  // Zustandsflags als Bitfeld.
  //
  // `Float32Array`, nicht `Uint8Array`: **WebGPU hat kein einkomponentiges
  // uint8-Vertexformat** — es gibt `uint8x2` und `uint8x4`, aber kein `x1`.
  // Ein `Uint8Array` mit itemSize 1 laesst die Geometrie still ausfallen (die
  // Punkte verschwinden komplett, ohne Fehlermeldung). `float32` ist dagegen
  // einkomponentig gueltig, und die Bitwerte bleiben als kleine Ganzzahlen
  // exakt darstellbar. Kosten: 400 KB bei 100k statt 100 KB — ein
  // `bufferSubData` darueber liegt weiterhin unter 0,2 ms.
  //
  // Die verbreitete Sorge, Attribute nicht neu hochladen zu duerfen, geht am
  // Problem vorbei: teuer waere es, Attributobjekte oder Geometrien neu zu
  // erzeugen, oder das pro Frame statt pro Eingabe zu tun.
  const flags = new Float32Array(n).fill(FLAG_VISIBLE);
  const flagAttribute = new InstancedBufferAttribute(flags, 1, false);
  flagAttribute.setUsage(DynamicDrawUsage);
  geometry.setAttribute('aFlags', flagAttribute);

  geometry.instanceCount = n;

  // Ausdehnung aus dem Manifest setzen, statt sie berechnen zu lassen: die
  // Positionsattribute enthalten das Template-Quad, nicht die Punktpositionen —
  // three wuerde eine Kugel mit Radius ~0,7 ermitteln. Sie darf auch nicht null
  // bleiben, three liest im Renderpfad `boundingSphere.center`.
  const extent = Math.max(
    ...bounds.min.map(Math.abs),
    ...bounds.max.map(Math.abs),
  );
  geometry.boundingBox = new Box3(new Vector3(...bounds.min), new Vector3(...bounds.max));
  geometry.boundingSphere = new Sphere(new Vector3(0, 0, 0), Math.max(bounds.radius, extent));
  geometry.computeBoundingSphere = () => {};
  geometry.computeBoundingBox = () => {};

  const uniforms = {
    offset: uniform(new Vector3(...dequant.offset)),
    half: uniform(new Vector3(...dequant.half)),
    // Weltgroesse eines Punktes, abgeleitet aus dem mittleren Punktabstand: so
    // bleibt die Darstellung bei jeder Korpusgroesse brauchbar, statt an einer
    // handgetunten Konstante zu haengen, die der naechste UMAP-Lauf entwertet.
    baseSize: uniform(meanSpacing(bounds.radius, n) * sizeFactor),
    // DER Regler gegen das Ruckeln: ohne ihn kosten die naechsten paar tausend
    // Punkte beim Hineinfliegen ein Vielfaches aller uebrigen zusammen.
    maxPx: uniform(maxPointPx),
    minPx: uniform(1),
    viewportHeight: uniform(1080),
    tanHalfFov: uniform(0.5),
    // Nicht-Treffer werden gedimmt, nicht versteckt: Verstecken zerstoert das
    // raeumliche Gedaechtnis, Dimmen erhaelt die Form als Kontext.
    dimAmount: uniform(0.88),
    searchActive: uniform(0),
    background: uniform(new Vector3(...background)),
  };

  // --- geteilte Bausteine ------------------------------------------------

  const flagBits = attribute('aFlags', 'float');
  /** Testet ein Flagbit auf einem als float uebergebenen Bitfeld. */
  const hasFlag = (bit) => floor(mod(flagBits.div(bit), 2.0));

  const vCorner = varying(attribute('position', 'vec3').xy, 'vCorner');
  const vState = varying(vec3(0), 'vState'); // x: sichtbar, y: hervorgehoben, z: id-hilfe

  /**
   * Weltposition, Blickraum, Pixelgroesse mit Clamp, Quad-Versatz.
   * Von Anzeige- und Picking-Material gemeinsam benutzt.
   */
  const buildVertex = Fn(([emphasisScale]) => {
    // Dequantisierung: exakt dieselbe Rechnung wie `dequantize_coords` in
    // artifacts.py.
    const world = attribute('aPos', 'vec4').xyz.mul(uniforms.half).add(uniforms.offset);
    const mv = modelViewMatrix.mul(vec4(world, 1.0));

    // Blickraum: die Kamera schaut entlang -z, die Distanz ist also -z.
    const dist = mv.z.negate().max(float(1e-4));
    const pxPerUnit = uniforms.viewportHeight.div(dist.mul(uniforms.tanHalfFov).mul(2.0));

    // Groesse in Pixeln bestimmen, dort clampen, dann zurueck in Blickraum-
    // Einheiten. Die Groessenabschwaechung bleibt als Tiefenhinweis erhalten;
    // nur die Explosion in der Naehe faellt weg.
    const px = uniforms.baseSize
      .mul(pxPerUnit)
      .clamp(uniforms.minPx, uniforms.maxPx)
      .mul(emphasisScale);
    const sizeView = px.div(pxPerUnit);

    const offset = attribute('position', 'vec3').xy.mul(sizeView);
    return cameraProjectionMatrix.mul(vec4(mv.xy.add(offset), mv.z, mv.w));
  });

  // --- Anzeige -----------------------------------------------------------

  const material = new MeshBasicNodeMaterial();
  // Opak und tiefengetestet statt additiv: korrekte Verdeckung, echte
  // Tiefenwahrnehmung, Early-Z in dichten Regionen und
  // Reihenfolgeunabhaengigkeit — alles gratis.
  material.transparent = false;
  material.depthTest = true;
  material.depthWrite = true;

  const vColor = varying(attribute('aColor', 'vec4').xyz, 'vColor');

  material.vertexNode = Fn(() => {
    const selected = hasFlag(FLAG_SELECTED);
    const neighbor = hasFlag(FLAG_NEIGHBOR);
    const match = hasFlag(FLAG_MATCH);

    // Hervorhebung vergroessert; Selektion am staerksten.
    const emphasis = float(1.0)
      .add(selected.mul(1.6))
      .add(neighbor.mul(0.6))
      .add(match.mul(uniforms.searchActive).mul(0.5));

    // Ein Punkt, der weder Filter noch Suche erfuellt, wird gedimmt — aber
    // NICHT versteckt. Echtes Verstecken passiert nur ueber `aFlags`-Bit 0,
    // und dann im Vertex-Shader durch Kollabieren auf Groesse null: ein
    // `discard` im Fragment kostet den vollen Fragment-Shader-Aufruf, ein
    // entartetes Quad erzeugt gar keine Fragmente.
    const visible = hasFlag(FLAG_VISIBLE);
    vState.assign(vec3(visible, selected.max(neighbor).max(match.mul(uniforms.searchActive)), 0));

    return buildVertex(emphasis.mul(visible));
  })();

  material.fragmentNode = Fn(() => {
    // 0 in der Mitte, 1 am Rand des eingeschriebenen Kreises.
    const d = vCorner.length().mul(2.0);
    Discard(d.greaterThan(1.0));

    // Dunkler Rand innerhalb des Sprites. Der Trick, der dichte Wolken lesbar
    // macht: ueberlappende Punkte trennen sich sichtbar, statt zu einer
    // einzigen Flaeche zu verschmelzen.
    //
    // Der Einsatzpunkt ist bewusst spaet: `d` ist ein Radius, die Flaeche
    // waechst quadratisch. Ein Rand ab 0,70 faerbt bereits die Haelfte des
    // Sprites ein (1 - 0,7^2 = 51 %); ab 0,82 sind es 33 %, der Kern bleibt farbig.
    const rim = smoothstep(0.82, 1.0, d);
    const base = mix(vColor, vColor.mul(0.35), rim);

    // Gedimmt wird gegen die Hintergrundfarbe, nicht gegen Schwarz: so
    // verschwinden Nicht-Treffer in der Tiefe, statt als dunkle Loecher vor
    // hellen Punkten zu stehen.
    const highlighted = vState.y;
    const dim = uniforms.searchActive.mul(float(1.0).sub(highlighted)).mul(uniforms.dimAmount);
    const shaded = mix(base, uniforms.background, dim);

    // Selektion und Nachbarn bekommen zusaetzlich einen hellen Ring.
    const ring = smoothstep(0.62, 0.80, d).mul(float(1.0).sub(smoothstep(0.92, 1.0, d)));
    return vec4(mix(shaded, vec3(1.0), ring.mul(highlighted).mul(0.75)), 1.0);
  })();

  // --- Picking -----------------------------------------------------------

  // Eigenes Material, gleicher Vertex-Knoten. Die ID wird im Vertex-Shader in
  // RGB kodiert und als Varying weitergereicht: `instanceIndex` steht im
  // Fragment-Shader unter WebGPU nicht zur Verfuegung.
  const vId = varying(vec3(0), 'vId');

  const pickingMaterial = new MeshBasicNodeMaterial();
  pickingMaterial.transparent = false;
  pickingMaterial.depthTest = true;
  pickingMaterial.depthWrite = true;

  pickingMaterial.vertexNode = Fn(() => {
    // +1, damit 0 der Hintergrund bleibt und nicht Punkt Nummer null ist.
    const id = float(instanceIndex).add(1.0);
    const r = mod(id, 256.0);
    const g = mod(floor(id.div(256.0)), 256.0);
    const b = mod(floor(id.div(65536.0)), 256.0);
    vId.assign(vec3(r, g, b).div(255.0));

    // Unsichtbare Punkte duerfen auch nicht anklickbar sein — sonst selektiert
    // man Dinge, die man nicht sieht. Genau dieser Fehler ist mit `Raycaster`
    // nicht behebbar, weil der von Filtern nichts weiss.
    return buildVertex(hasFlag(FLAG_VISIBLE));
  })();

  pickingMaterial.fragmentNode = Fn(() => {
    const d = vCorner.length().mul(2.0);
    Discard(d.greaterThan(1.0));
    return vec4(vId, 1.0);
  })();

  // --- Mesh --------------------------------------------------------------

  const mesh = new Mesh(geometry, material);
  mesh.frustumCulled = false; // eine Geometrie, die immer sichtbar ist

  mesh.uniforms = uniforms;
  mesh.pickingMaterial = pickingMaterial;
  mesh.displayMaterial = material;
  mesh.flags = flags;
  mesh.flagAttribute = flagAttribute;

  /** Muss pro Frame aufgerufen werden — der Pixel-Clamp haengt an beidem. */
  mesh.updateScreenMetrics = (camera, drawingBufferHeight) => {
    uniforms.viewportHeight.value = drawingBufferHeight;
    uniforms.tanHalfFov.value = Math.tan((camera.fov * Math.PI) / 360);
  };

  /** Flags zur GPU schieben. Pro Eingabe-Event, nicht pro Frame. */
  mesh.commitFlags = () => {
    flagAttribute.needsUpdate = true;
  };

  return mesh;
}

/** Grober mittlerer Punktabstand in einer Kugel mit Radius `r`. */
function meanSpacing(radius, n) {
  if (!(radius > 0) || !(n > 0)) return 1;
  return (2 * radius) / Math.cbrt(n);
}
