/**
 * Die Punktwolke: eine Geometrie, ein Draw Call, instanzierte Quads.
 *
 * Bewusst NICHT `THREE.Points` / `gl.POINTS`, aus vier unabhaengigen Gruenden
 * (siehe docs/decisions.md E3):
 *
 *  1. `gl_PointSize` ist treiberseitig gedeckelt — auf Apple Silicon bei 64 px.
 *  2. `gl.POINTS` clippt am Mittelpunkt, nicht an der Ausdehnung; grosse Sprites
 *     verschwinden schlagartig am Bildrand.
 *  3. WebGPU hat ueberhaupt keine Punktgroesse: `point-list` rendert 1x1 Pixel.
 *  4. Rundung, dunkler Rand und Groessen-Clamp brauchen ohnehin Kontrolle pro
 *     Fragment.
 *
 * Kosten: vier statt einem Vertex pro Punkt. Bei 1 Mio. Punkten sind das ~0,3 ms
 * Vertex-Arbeit auf Mittelklasse-Hardware — irrelevant, weil der Engpass die
 * Fuellrate ist, nicht die Vertexzahl.
 */

import {
  Box3,
  BufferAttribute,
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
  mix,
  modelViewMatrix,
  smoothstep,
  uniform,
  varying,
  vec4,
} from 'three/tsl';

/**
 * Baut die Punktwolke aus einem geladenen Artefaktsatz.
 *
 * @param {object} data Rueckgabe von `loadArtifacts`.
 * @param {object} [options]
 * @param {number} [options.maxPointPx] Obergrenze der Bildschirmgroesse in Pixeln.
 * @param {number} [options.sizeFactor] Punktgroesse als Bruchteil des mittleren
 *   Punktabstands.
 */
export function createPointCloud(data, options = {}) {
  const { maxPointPx = 48, sizeFactor = 0.25 } = options;
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
  geometry.instanceCount = n;

  // Ausdehnung aus dem Manifest setzen, statt sie berechnen zu lassen: die
  // Positionsattribute enthalten das Template-Quad, nicht die Punktpositionen —
  // three wuerde also eine Kugel mit Radius ~0.7 ermitteln. Sie darf auch nicht
  // null bleiben, three liest im Renderpfad `boundingSphere.center`.
  const extent = Math.max(
    Math.abs(bounds.min[0]), Math.abs(bounds.max[0]),
    Math.abs(bounds.min[1]), Math.abs(bounds.max[1]),
    Math.abs(bounds.min[2]), Math.abs(bounds.max[2]),
  );
  geometry.boundingBox = new Box3(
    new Vector3(...bounds.min),
    new Vector3(...bounds.max),
  );
  geometry.boundingSphere = new Sphere(new Vector3(0, 0, 0), Math.max(bounds.radius, extent));
  geometry.computeBoundingSphere = () => {};
  geometry.computeBoundingBox = () => {};

  const uniforms = {
    offset: uniform(new Vector3(...dequant.offset)),
    half: uniform(new Vector3(...dequant.half)),
    // Weltgroesse eines Punktes, abgeleitet aus dem mittleren Punktabstand:
    // so bleibt die Darstellung bei jeder Korpusgroesse brauchbar, statt an
    // einer handgetunten Konstante zu haengen, die der naechste UMAP-Lauf
    // entwertet.
    baseSize: uniform(meanSpacing(bounds.radius, n) * sizeFactor),
    // DER Regler gegen das Ruckeln. Ohne ihn kosten die naechsten paar tausend
    // Punkte beim Hineinfliegen ein Vielfaches aller uebrigen zusammen.
    maxPx: uniform(maxPointPx),
    minPx: uniform(1),
    viewportHeight: uniform(1080),
    tanHalfFov: uniform(0.5),
  };

  const material = new MeshBasicNodeMaterial();
  // Opak und tiefengetestet statt additiv: korrekte Verdeckung, echte
  // Tiefenwahrnehmung, Early-Z in dichten Regionen und
  // Reihenfolgeunabhaengigkeit — alles gratis. Additiv bleibt ein moeglicher
  // Zusatzmodus, ist als Vorgabe aber falsch.
  material.transparent = false;
  material.depthTest = true;
  material.depthWrite = true;

  // Interpolierte Quad-Koordinate: im Fragment brauchen wir den Abstand zur
  // Sprite-Mitte fuer Rundung und Rand.
  const vCorner = varying(attribute('position', 'vec3').xy, 'vCorner');
  const vColor = varying(attribute('aColor', 'vec4').xyz, 'vColor');

  material.vertexNode = Fn(() => {
    // Dequantisierung: exakt dieselbe Rechnung wie `dequantize_coords` in
    // artifacts.py.
    const world = attribute('aPos', 'vec4').xyz.mul(uniforms.half).add(uniforms.offset);
    const mv = modelViewMatrix.mul(vec4(world, 1.0));

    // Blickraum: die Kamera schaut entlang -z, die Distanz ist also -z.
    const dist = mv.z.negate().max(float(1e-4));
    const pxPerUnit = uniforms.viewportHeight.div(dist.mul(uniforms.tanHalfFov).mul(2.0));

    // Groesse in Pixeln bestimmen, dort clampen, dann zurueck in Blickraum-
    // Einheiten. Die Groessenabschwaechung bleibt damit als Tiefenhinweis
    // erhalten — nur die Explosion in der Naehe faellt weg.
    const px = uniforms.baseSize.mul(pxPerUnit).clamp(uniforms.minPx, uniforms.maxPx);
    const sizeView = px.div(pxPerUnit);

    const offset = attribute('position', 'vec3').xy.mul(sizeView);
    return cameraProjectionMatrix.mul(vec4(mv.xy.add(offset), mv.z, mv.w));
  })();

  material.fragmentNode = Fn(() => {
    // 0 in der Mitte, 1 am Rand des eingeschriebenen Kreises.
    const d = vCorner.length().mul(2.0);
    Discard(d.greaterThan(1.0));

    // Dunkler Rand innerhalb des Sprites. Der Trick, der dichte Wolken lesbar
    // macht: ueberlappende Punkte trennen sich sichtbar, statt zu einer
    // einzigen Flaeche zu verschmelzen.
    //
    // Der Einsatzpunkt ist bewusst spaet: `d` ist ein Radius, die Flaeche waechst
    // quadratisch. Ein Rand ab 0,70 faerbt bereits die Haelfte des Sprites ein
    // (1 - 0,7^2 = 51 %) und laesst kleine Punkte durchgehend dunkel wirken.
    // Ab 0,82 sind es 33 %, der Kern bleibt farbig.
    const rim = smoothstep(0.82, 1.0, d);
    return vec4(mix(vColor, vColor.mul(0.35), rim), 1.0);
  })();

  const mesh = new Mesh(geometry, material);
  mesh.frustumCulled = false; // eine Geometrie, die immer sichtbar ist

  /** Muss pro Frame aufgerufen werden — der Pixel-Clamp haengt an beidem. */
  mesh.updateScreenMetrics = (camera, drawingBufferHeight) => {
    uniforms.viewportHeight.value = drawingBufferHeight;
    uniforms.tanHalfFov.value = Math.tan((camera.fov * Math.PI) / 360);
  };

  mesh.uniforms = uniforms;
  return mesh;
}

/** Grober mittlerer Punktabstand in einer Kugel mit Radius `r`. */
function meanSpacing(radius, n) {
  if (!(radius > 0) || !(n > 0)) return 1;
  return (2 * radius) / Math.cbrt(n);
}
