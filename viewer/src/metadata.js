/**
 * Metadaten im Browser: Facetten dicht, Text als ein Block.
 *
 * Gegenstueck zu `ArtifactWriter.add_metadata`. Zwei Formate, weil die
 * Zugriffsmuster verschieden sind:
 *
 * * **Facetten** werden bei jedem Filterklick ueber alle Punkte gelesen und
 *   liegen als dichtes `uint8`-Array vor.
 * * **Text** liegt als ein einziger UTF-8-Block vor, dessen Datensaetze durch
 *   ASCII 30 getrennt sind. Er wird **einmal** dekodiert und dient danach
 *   sowohl dem Detailpanel als auch als Suchindex.
 *
 * Der Unterschied ist bei Zielgroesse erheblich. Jeden Datensatz einzeln zu
 * dekodieren heisst bei 100.000 Punkten 100.000 Aufrufe von `TextDecoder` plus
 * 100.000 `split`-Aufrufe plus 100.000 kurzlebige Stringobjekte. Ein einziger
 * Aufruf ueber den ganzen Block kostet einen Bruchteil davon, und die Suche
 * wird danach zu einem `indexOf` ueber einen zusammenhaengenden String.
 *
 * Deshalb enthaelt der Block auch **keine URLs**: er ist zugleich der
 * Suchindex, und jede Met-URL enthaelt "/art/collection/search/" — eine Suche
 * nach "art" wuerde sonst jeden Datensatz treffen. Der Link wird aus der
 * Objekt-ID zusammengesetzt.
 */

const decoder = new TextDecoder('utf-8');

/** Aus dieser ID baut der Viewer die Sammlungs-URL. */
export const MET_OBJECT_URL = 'https://www.metmuseum.org/art/collection/search/';

export class Metadata {
  constructor({
    n, fields, separator, recordSeparator, facetFields, facetValues, facets, text, offsets,
  }) {
    this.n = n;
    this.fields = fields;
    this.separator = separator;
    this.recordSeparator = recordSeparator ?? '';
    this.facetFields = facetFields;
    this.facetValues = facetValues;
    this.facets = facets;
    this.offsets = offsets;

    // Einmal dekodieren. Danach wird nur noch in diesem String gearbeitet.
    this.blob = text ? decoder.decode(text) : '';
    // Kleingeschrieben fuer die Suche — ebenfalls genau einmal.
    this.haystack = this.blob.toLowerCase();

    // Zeichen-Offsets der Datensaetze. Die Byte-Offsets aus der Datei taugen
    // dafuer nicht: JS-Strings zaehlen UTF-16-Einheiten, und bei Umlauten,
    // CJK-Zeichen oder Emoji laufen beide auseinander. Deshalb einmal ueber
    // den Trenner scannen.
    this.starts = new Uint32Array(n + 1);
    let position = 0;
    for (let index = 0; index < n; index++) {
      this.starts[index] = position;
      const next = this.blob.indexOf(this.recordSeparator, position);
      position = next === -1 ? this.blob.length : next + 1;
    }
    this.starts[n] = this.blob.length;
  }

  /** Facettenwert eines Punktes als Text. */
  facet(index, field) {
    const column = this.facetFields.indexOf(field);
    if (column < 0) return '';
    const value = this.facets[index * this.facetFields.length + column];
    return this.facetValues[field][value] ?? '';
  }

  /** Facettenindex eines Punktes — fuer Filtervergleiche ohne Stringarbeit. */
  facetIndex(index, column) {
    return this.facets[index * this.facetFields.length + column];
  }

  /** Rohtext eines Datensatzes, ohne den abschliessenden Trenner. */
  raw(index) {
    if (index < 0 || index >= this.n) return '';
    const end = Math.max(this.starts[index], this.starts[index + 1] - 1);
    return this.blob.slice(this.starts[index], end);
  }

  /** Alle Textfelder eines Punktes als Objekt, plus abgeleiteter Link. */
  record(index) {
    if (!this.blob || index < 0 || index >= this.n) return null;
    const parts = this.raw(index).split(this.separator);
    const out = {};
    this.fields.forEach((name, position) => {
      out[name] = parts[position] ?? '';
    });
    if (out.object_id) out.link = MET_OBJECT_URL + out.object_id;
    return out;
  }

  title(index) {
    return this.raw(index).split(this.separator)[0] ?? '';
  }

  /**
   * Indizes aller Punkte, deren Text `query` enthaelt.
   *
   * Ein `indexOf` ueber den ganzen Block statt einer Schleife ueber n
   * Einzelstrings: bei 100.000 Datensaetzen ist das der Unterschied zwischen
   * einigen Millisekunden und einem sichtbaren Hänger.
   */
  search(query, limit = 5000) {
    const needle = query.trim().toLowerCase();
    if (!needle || !this.haystack) return [];

    const hits = [];
    let position = this.haystack.indexOf(needle);
    let last = -1;
    while (position !== -1 && hits.length < limit) {
      const index = this.recordAt(position);
      // Mehrere Treffer im selben Datensatz zaehlen einmal.
      if (index !== last) {
        hits.push(index);
        last = index;
      }
      // Ab dem Ende des aktuellen Datensatzes weitersuchen, wenn er schon
      // getroffen hat — sonst laufen wir bei haeufigen Woertern durch jeden
      // einzelnen Treffer desselben Datensatzes.
      position = this.haystack.indexOf(needle, Math.max(position + 1, this.starts[index + 1]));
    }
    return hits;
  }

  /** Datensatzindex zu einer Position im Block, per binaerer Suche. */
  recordAt(position) {
    let low = 0;
    let high = this.n - 1;
    while (low < high) {
      const middle = (low + high + 1) >> 1;
      if (this.starts[middle] <= position) low = middle;
      else high = middle - 1;
    }
    return low;
  }
}
