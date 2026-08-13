/**
 * Die Browserseite des Datenvertrags.
 *
 * Gegenstueck zu `pipeline/src/lse/artifacts.py`. Die beiden Dateien beschreiben
 * dasselbe Format aus zwei Richtungen und gehoeren in dasselbe Review — wer eine
 * aendert, muss die andere mitaendern.
 *
 * Alles Binaere wird kopierfrei gelesen: `new Int16Array(buffer)` uebernimmt den
 * Puffer direkt. Deshalb sind die Dateien headerlos und explizit little-endian,
 * und deshalb sind Koordinaten und Farben vierkomponentig (WebGPU kennt keine
 * dreikomponentigen 8/16-Bit-Vertexformate).
 */

/** Muss zu `SCHEMA_VERSION` in `pipeline/src/lse/config.py` passen. */
export const SCHEMA_VERSION = 1;

export class ArtifactError extends Error {}

async function fetchBinary(base, name, expectedBytes) {
  const response = await fetch(`${base}/${name}`);
  if (!response.ok) {
    throw new ArtifactError(`${name}: HTTP ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  if (expectedBytes !== undefined && buffer.byteLength !== expectedBytes) {
    throw new ArtifactError(
      `${name}: ${buffer.byteLength} Bytes gelesen, ${expectedBytes} laut Manifest. ` +
        'Pipeline und Viewer sind auseinandergelaufen.',
    );
  }
  return buffer;
}

/**
 * Laedt einen Artefaktsatz.
 *
 * @param {string} base Basis-URL des Verzeichnisses, z. B. `/data/sample`.
 * @returns {Promise<{
 *   manifest: object, n: number, coords: Int16Array, colors: Uint8Array,
 *   labels: Uint16Array, labelNames: string[],
 *   loadNeighbors: () => Promise<Uint32Array|null>, neighborsK: number,
 *   dequant: {offset: number[], half: number[]},
 *   bounds: {min: number[], max: number[], radius: number},
 *   strides: {coord: number, color: number},
 * }>}
 */
export async function loadArtifacts(base) {
  const response = await fetch(`${base}/manifest.json`);
  if (!response.ok) {
    throw new ArtifactError(
      `Kein Manifest unter ${base} (HTTP ${response.status}). ` +
        'Erzeugen mit: uv run --project pipeline lse synth --n 1000 --out data/sample',
    );
  }
  const manifest = await response.json();

  if (manifest.schema_version !== SCHEMA_VERSION) {
    throw new ArtifactError(
      `schema_version ${manifest.schema_version}, Viewer erwartet ${SCHEMA_VERSION}.`,
    );
  }

  const n = manifest.n;
  const files = Object.fromEntries(manifest.files.map((f) => [f.name, f]));
  const coordStride = manifest.coords.coord_stride;
  const colorStride = manifest.coords.color_stride;

  const [coordsBuf, colorsBuf, labelsBuf] = await Promise.all([
    fetchBinary(base, 'coords.i16.bin', files['coords.i16.bin']?.bytes),
    fetchBinary(base, 'colors.u8.bin', files['colors.u8.bin']?.bytes),
    fetchBinary(base, 'labels.u16.bin', files['labels.u16.bin']?.bytes),
  ]);

  // Nachbarn werden NICHT mitgeladen. Sie tragen die Ehrlichkeitsschicht (die
  // echten hochdimensionalen Nachbarn eines Punktes) und werden erst beim ersten
  // Klick gebraucht — sind aber die mit Abstand groesste Datei: 20 x uint32 x N
  // sind 8 MB bei 100k, gegenueber 1,4 MB fuer alles Renderkritische zusammen.
  // Blockierend geladen waere das eine Versechsfachung der Zeit bis zum ersten
  // Bild, fuer eine Funktion, die der Nutzer vielleicht nie ausloest.
  const neighborEntry = files['neighbors.u32.bin'];
  let neighborsPromise = null;
  const loadNeighbors = () => {
    if (!neighborEntry) return Promise.resolve(null);
    neighborsPromise ??= fetchBinary(base, 'neighbors.u32.bin', neighborEntry.bytes).then(
      (buffer) => new Uint32Array(buffer),
    );
    return neighborsPromise;
  };

  // Facetten sind klein und werden bei jedem Filterklick gebraucht: bei 100k
  // Punkten und vier Facetten 400 KB. Die laden wir mit.
  let facets = null;
  const facetEntry = files['facets.u8.bin'];
  if (facetEntry) {
    facets = new Uint8Array(await fetchBinary(base, 'facets.u8.bin', facetEntry.bytes));
  }

  const coords = new Int16Array(coordsBuf);
  const colors = new Uint8Array(colorsBuf);
  const labels = new Uint16Array(labelsBuf);

  if (coords.length !== n * coordStride) {
    throw new ArtifactError(`coords: ${coords.length} Werte, erwartet ${n * coordStride}.`);
  }
  if (colors.length !== n * colorStride) {
    throw new ArtifactError(`colors: ${colors.length} Werte, erwartet ${n * colorStride}.`);
  }
  if (labels.length !== n) {
    throw new ArtifactError(`labels: ${labels.length} Werte, erwartet ${n}.`);
  }

  // Der Textblock traegt Titel, Kuenstler, Datum und Link aller Punkte. Er wird
  // fuer das erste Bild nicht gebraucht und deshalb nachgeladen — bei 100k
  // Punkten sind das mehrere MB gegen 1,4 MB fuer alles Renderkritische.
  const textEntry = files['text.bin'];
  const offsetsEntry = files['text_offsets.u32.bin'];
  let metadataPromise = null;
  const loadMetadata = () => {
    if (!facets || !textEntry || !offsetsEntry || !manifest.metadata?.facet_fields) {
      return Promise.resolve(null);
    }
    metadataPromise ??= Promise.all([
      fetchBinary(base, 'text.bin', textEntry.bytes),
      fetchBinary(base, 'text_offsets.u32.bin', offsetsEntry.bytes),
    ]).then(async ([textBuf, offsetBuf]) => {
      const { Metadata } = await import('./metadata.js');
      return new Metadata({
        n,
        fields: manifest.metadata.text_fields,
        separator: manifest.metadata.field_separator,
        recordSeparator: manifest.metadata.record_separator,
        facetFields: manifest.metadata.facet_fields,
        facetValues: manifest.metadata.facet_values,
        facets,
        text: new Uint8Array(textBuf),
        offsets: new Uint32Array(offsetBuf),
      });
    });
    return metadataPromise;
  };

  return {
    manifest,
    n,
    coords,
    colors,
    labels,
    facets,
    labelNames: manifest.corpus.label_names ?? [],
    loadNeighbors,
    loadMetadata,
    neighborsK: neighborEntry?.shape[1] ?? 0,
    dequant: { offset: manifest.coords.offset, half: manifest.coords.half },
    bounds: {
      min: manifest.coords.min,
      max: manifest.coords.max,
      radius: manifest.coords.radius,
    },
    strides: { coord: coordStride, color: colorStride },
  };
}
