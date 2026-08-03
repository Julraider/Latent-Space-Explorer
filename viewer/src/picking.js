/**
 * GPU-ID-Picking.
 *
 * `Raycaster` gegen eine Punktwolke ist nicht nur langsam, sondern **falsch**.
 * Aus dem three-Quellcode (`Points.raycast`): er iteriert linear ueber jeden
 * Vertex, und sein Schwellwert ist ein konstanter Weltraum-Radius, der
 * `PointsMaterial.size` und jedes eigene `gl_PointSize` schlicht ignoriert. Ein
 * Punkt zwei Einheiten entfernt, der als 60-px-Scheibe das Bild fuellt, bekommt
 * damit dieselbe Trefferflaeche wie einer 40 Einheiten entfernt, der 2 px gross
 * ist — man klickt auf den grossen und selektiert etwas dahinter. Von Filtern
 * weiss er ebenfalls nichts, unsichtbare Punkte bleiben also anklickbar.
 *
 * Der ID-Pass benutzt **denselben Vertex-Shader wie die Anzeige**. Damit stimmt
 * die Trefferflaeche automatisch mit Groessenabschwaechung, Groessen-Clamp,
 * runder Maske, Tiefenverdeckung und jedem Filterzustand ueberein — ohne eine
 * zweite Kopie dieser Logik, die beim naechsten Shader-Tuning auseinanderdriftet.
 */

import { NearestFilter, RenderTarget, Vector4 } from 'three/webgpu';

// Es wird eine kleine Region um den Cursor gelesen, nicht ein einzelnes Pixel.
// Kostet dasselbe und liefert Snap-to-Nearest: niemand muss einen 3-px-Punkt
// exakt treffen.
const REGION = 11;

export class Picker {
  constructor(renderer, scene, camera, cloud) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.cloud = cloud;

    this.target = new RenderTarget(REGION, REGION);
    this.target.texture.minFilter = NearestFilter;
    this.target.texture.magFilter = NearestFilter;
    this.target.texture.generateMipmaps = false;

    this._viewport = new Vector4();
    this._busy = false;
  }

  /**
   * Liefert den Punktindex unter (x, y) in CSS-Pixeln, oder -1.
   *
   * Asynchron: `readRenderTargetPixelsAsync` vermeidet den synchronen Stall,
   * der beim Hover sonst permanent auf die GPU wartet und die Bildrate halbiert.
   * Ueberlappende Aufrufe werden verworfen statt eingereiht — beim Hover ist
   * die neueste Position die einzige, die zaehlt.
   */
  async pick(x, y) {
    if (this._busy) return -1;
    this._busy = true;
    try {
      const { renderer, camera, cloud, scene } = this;
      const size = renderer.getSize(this._viewport);
      const pixelRatio = renderer.getPixelRatio();

      const half = (REGION - 1) / 2;
      // Nur den Ausschnitt um den Cursor rendern statt des ganzen Bildes: die
      // Kamera projiziert dabei denselben Raum, nur auf REGION x REGION Pixel.
      camera.setViewOffset(
        size.x * pixelRatio,
        size.y * pixelRatio,
        Math.round(x * pixelRatio) - half,
        Math.round(y * pixelRatio) - half,
        REGION,
        REGION,
      );

      const previousMaterial = cloud.material;
      const previousBackground = scene.background;
      cloud.material = cloud.pickingMaterial;
      // Hintergrund und Nebel wuerden die ID-Farben verfaelschen.
      scene.background = null;
      const previousFog = scene.fog;
      scene.fog = null;

      renderer.setRenderTarget(this.target);
      renderer.setClearColor(0x000000, 0);
      renderer.clear();
      await renderer.renderAsync(scene, camera);

      const pixels = await renderer.readRenderTargetPixelsAsync(
        this.target, 0, 0, REGION, REGION,
      );

      renderer.setRenderTarget(null);
      cloud.material = previousMaterial;
      scene.background = previousBackground;
      scene.fog = previousFog;
      camera.clearViewOffset();

      return decodeNearest(pixels, REGION);
    } finally {
      this._busy = false;
    }
  }

  dispose() {
    this.target.dispose();
  }
}

/**
 * Sucht in der gelesenen Region die ID, die der Mitte am naechsten liegt.
 *
 * Nicht einfach das mittlere Pixel: bei duennen Punkten trifft man sonst fast
 * immer den Hintergrund, und der Klick fuehlt sich kaputt an.
 */
function decodeNearest(pixels, region) {
  const center = (region - 1) / 2;
  let best = -1;
  let bestDistance = Infinity;

  for (let row = 0; row < region; row++) {
    for (let column = 0; column < region; column++) {
      const offset = (row * region + column) * 4;
      const id = pixels[offset] | (pixels[offset + 1] << 8) | (pixels[offset + 2] << 16);
      if (id === 0) continue; // Hintergrund

      // Die Textur steht gegenueber den Fensterkoordinaten auf dem Kopf; fuer
      // die Distanz zur Mitte ist das egal, fuer die Symmetrie nicht.
      const dx = column - center;
      const dy = row - center;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = id - 1; // die +1 aus dem Shader zuruecknehmen
      }
    }
  }
  return best;
}
