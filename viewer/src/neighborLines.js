/**
 * Die Ehrlichkeitsschicht.
 *
 * UMAP-Abstaende sind nicht bedeutsam. Nur lokale Nachbarschaften sind es, und
 * selbst die werden verzerrt. Ein Raum, der so tut, als waere Naehe im Bild
 * gleich Naehe im Modell, luegt — nur eben hübsch.
 *
 * Deshalb werden bei jeder Auswahl die **echten hochdimensionalen** Nachbarn
 * eines Punktes verbunden, vorberechnet in der Pipeline. Wo eine Linie quer
 * durch die Wolke laeuft, sieht man unmittelbar, was die Projektion
 * weggeworfen hat. Das kostet einen Build-Schritt und ein `LineSegments` — und
 * es ist der einzige Teil der Darstellung, der etwas zeigt, das ein
 * Standard-Scatterplot nicht kann.
 *
 * Die Linien sind eine Anmerkung, keine Geometrie: sie ignorieren den
 * Tiefentest bewusst, sonst waeren genau die interessanten (langen, quer durch
 * dichte Regionen laufenden) Verbindungen unsichtbar.
 */

import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  LineBasicNodeMaterial,
  LineSegments,
  Sphere,
  Vector3,
} from 'three/webgpu';
import { attribute, varying, vec4 } from 'three/tsl';

export function createNeighborLines(maxNeighbors, cloudRadius = 1) {
  const vertexCount = maxNeighbors * 2;
  const geometry = new BufferGeometry();

  const positions = new Float32Array(vertexCount * 3);
  const intensities = new Float32Array(vertexCount);
  geometry.setAttribute('position', new BufferAttribute(positions, 3));
  geometry.setAttribute('aIntensity', new BufferAttribute(intensities, 1));
  geometry.setDrawRange(0, 0);

  // Die Bounding-Sphere muss gesetzt sein und darf NICHT null bleiben: three
  // liest im Renderpfad `boundingSphere.center`, und ein null-Wert wirft dort
  // pro Frame. Berechnen lassen geht ebenfalls nicht — die Puffer sind
  // anfangs leer und werden danach laufend ueberschrieben. Also einmal die
  // Wolke umschliessen und dabei belassen; `frustumCulled = false` sorgt
  // ohnehin dafuer, dass nichts faelschlich verworfen wird.
  geometry.boundingSphere = new Sphere(new Vector3(0, 0, 0), cloudRadius * 2);
  geometry.computeBoundingSphere = () => {};

  const material = new LineBasicNodeMaterial();
  material.transparent = true;
  // Additiv: die Linien sollen ueber der Wolke leuchten, nicht sie verdecken.
  material.blending = AdditiveBlending;
  // Anmerkung, keine Geometrie — sonst verschwinden genau die langen
  // Verbindungen, die die Verzerrung zeigen, im Inneren der Wolke.
  material.depthTest = false;
  material.depthWrite = false;

  const vIntensity = varying(attribute('aIntensity', 'float'), 'vIntensity');
  material.fragmentNode = vec4(0.62, 0.80, 1.0, vIntensity.mul(0.85));

  const lines = new LineSegments(geometry, material);
  lines.frustumCulled = false;
  lines.renderOrder = 10;

  /**
   * Setzt die Linien neu.
   *
   * @param {number[]} origin Weltposition des ausgewaehlten Punktes.
   * @param {number[][]} targets Weltpositionen der Nachbarn, in Rangfolge.
   */
  lines.setNeighbors = (origin, targets) => {
    const count = Math.min(targets.length, maxNeighbors);
    for (let i = 0; i < count; i++) {
      const base = i * 6;
      positions[base] = origin[0];
      positions[base + 1] = origin[1];
      positions[base + 2] = origin[2];
      positions[base + 3] = targets[i][0];
      positions[base + 4] = targets[i][1];
      positions[base + 5] = targets[i][2];

      // Naehere Raenge heller: die Rangfolge ist eine Information, und ohne
      // sie sehen 20 gleich helle Linien wie ein Stern aus statt wie eine
      // Nachbarschaft.
      const rank = 1 - i / Math.max(count - 1, 1);
      intensities[i * 2] = 1.0;
      intensities[i * 2 + 1] = 0.25 + 0.6 * rank;
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.aIntensity.needsUpdate = true;
    geometry.setDrawRange(0, count * 2);
    lines.visible = count > 0;
  };

  lines.clear = () => {
    geometry.setDrawRange(0, 0);
    lines.visible = false;
  };

  lines.visible = false;
  return lines;
}
