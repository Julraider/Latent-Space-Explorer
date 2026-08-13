/**
 * Der Marker fuer einen live platzierten Prompt.
 *
 * Optisch bewusst anders als die Korpuspunkte: ein pulsierender Ring statt
 * einer gefuellten Scheibe. Er muss auf den ersten Blick als "meine Anfrage"
 * lesbar sein und nicht als Objekt aus der Sammlung — sonst behauptet die
 * Darstellung, im Korpus gaebe es etwas, das es nicht gibt.
 *
 * Eine einzelne Instanz, deshalb Position als Uniform statt als Attribut: das
 * spart den Umweg ueber einen Puffer, der bei jeder Anfrage neu geschrieben
 * werden muesste.
 */

import { Mesh, MeshBasicNodeMaterial, PlaneGeometry, Sphere, Vector3 } from 'three/webgpu';
import {
  Discard,
  Fn,
  cameraProjectionMatrix,
  float,
  modelViewMatrix,
  positionGeometry,
  sin,
  smoothstep,
  time,
  uniform,
  varying,
  vec3,
  vec4,
} from 'three/tsl';

export function createPromptMarker({ maxPointPx = 96, worldSize = 1 } = {}) {
  const geometry = new PlaneGeometry(1, 1);
  // Nicht null lassen: three liest im Renderpfad `boundingSphere.center`.
  geometry.boundingSphere = new Sphere(new Vector3(0, 0, 0), 1);
  geometry.computeBoundingSphere = () => {};

  const uniforms = {
    position: uniform(new Vector3(0, 0, 0)),
    baseSize: uniform(worldSize),
    maxPx: uniform(maxPointPx),
    minPx: uniform(14),
    viewportHeight: uniform(1080),
    tanHalfFov: uniform(0.5),
    // Ein unsicherer Treffer wird warm eingefaerbt statt still akzeptiert:
    // wenn die naechsten Nachbarn im Raum weit auseinanderliegen, sitzt der
    // Punkt zwischen Regionen und moeglicherweise in einer Leerstelle.
    warn: uniform(0),
  };

  const material = new MeshBasicNodeMaterial();
  material.transparent = true;
  // Immer sichtbar: der Marker ist eine Antwort auf eine Nutzeraktion, kein
  // Korpusobjekt. Ihn hinter der Wolke verschwinden zu lassen liest sich als
  // "hat nicht funktioniert".
  material.depthTest = false;
  material.depthWrite = false;

  const vCorner = varying(positionGeometry.xy, 'vMarkerCorner');

  material.vertexNode = Fn(() => {
    const mv = modelViewMatrix.mul(vec4(uniforms.position, 1.0));
    const dist = mv.z.negate().max(float(1e-4));
    const pxPerUnit = uniforms.viewportHeight.div(dist.mul(uniforms.tanHalfFov).mul(2.0));

    // Derselbe Groessen-Clamp wie bei den Punkten, nur mit hoeherer Obergrenze:
    // der Marker soll auch aus der Uebersicht sichtbar bleiben.
    const px = uniforms.baseSize.mul(pxPerUnit).clamp(uniforms.minPx, uniforms.maxPx);
    const sizeView = px.div(pxPerUnit);

    const offset = positionGeometry.xy.mul(sizeView);
    return cameraProjectionMatrix.mul(vec4(mv.xy.add(offset), mv.z, mv.w));
  })();

  material.fragmentNode = Fn(() => {
    const d = vCorner.length().mul(2.0);
    Discard(d.greaterThan(1.0));

    // Pulsierender Ring. Die Bewegung ist der eigentliche Hinweis: ein
    // statischer Ring geht in einer Wolke aus Kreisen unter.
    const pulse = sin(time.mul(3.0)).mul(0.5).add(0.5);
    const inner = float(0.52).add(pulse.mul(0.12));
    const ring = smoothstep(inner, inner.add(0.10), d).mul(
      float(1.0).sub(smoothstep(0.86, 1.0, d)),
    );
    const core = float(1.0).sub(smoothstep(0.0, 0.22, d));

    const good = vec3(1.0, 1.0, 1.0);
    const warn = vec3(1.0, 0.72, 0.42);
    const tint = good.mix(warn, uniforms.warn);

    return vec4(tint, ring.mul(0.95).add(core.mul(0.9)));
  })();

  const marker = new Mesh(geometry, material);
  marker.frustumCulled = false;
  marker.renderOrder = 20;
  marker.visible = false;
  marker.uniforms = uniforms;

  marker.setPosition = (position, { warn = false } = {}) => {
    uniforms.position.value.set(position[0], position[1], position[2]);
    uniforms.warn.value = warn ? 1 : 0;
    marker.visible = true;
  };

  marker.hide = () => {
    marker.visible = false;
  };

  marker.updateScreenMetrics = (camera, drawingBufferHeight) => {
    uniforms.viewportHeight.value = drawingBufferHeight;
    uniforms.tanHalfFov.value = Math.tan((camera.fov * Math.PI) / 360);
  };

  return marker;
}
