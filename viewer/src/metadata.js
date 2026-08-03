/**
 * Metadaten im Browser: Facetten dicht, Text als ein Block.
 *
 * Gegenstueck zu `ArtifactWriter.add_metadata`. Zwei Formate, weil die
 * Zugriffsmuster verschieden sind:
 *
 * * **Facetten** werden bei jedem Filterklick ueber alle Punkte gelesen und
 *   liegen als dichtes `uint8`-Array vor.
 * * **Text** wird immer nur fuer einen einzelnen Punkt gebraucht und liegt als
 *   ein UTF-8-Block mit Offsettabelle vor. 100.000 JSON-Objekte zu parsen
 *   kostet hunderte Millisekunden Hauptthread; ein Block plus Offsets kostet
 *   nichts.
 */

const decoder = new TextDecoder('utf-8');

export class Metadata {
  constructor({ n, fields, separator, facetFields, facetValues, facets, text, offsets }) {
    this.n = n;
    this.fields = fields;
    this.separator = separator;
    this.facetFields = facetFields;
    this.facetValues = facetValues;
    this.facets = facets; // Uint8Array [n * facetFields.length]
    this.text = text; // Uint8Array oder null, solange nicht geladen
    this.offsets = offsets; // Uint32Array [n + 1]
    this._titles = null;
    this._searchIndex = null;
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

  /** Alle Textfelder eines Punktes als Objekt. */
  record(index) {
    if (!this.text) return null;
    const slice = this.text.subarray(this.offsets[index], this.offsets[index + 1]);
    const parts = decoder.decode(slice).split(this.separator);
    const out = {};
    this.fields.forEach((name, position) => {
      out[name] = parts[position] ?? '';
    });
    return out;
  }

  /**
   * Titel aller Punkte, kleingeschrieben, einmalig aufgebaut.
   *
   * Bewusst als ein einziger konkatenierter String mit Offsets statt als Array
   * von n Strings: bei 100k Punkten spart das 100.000 kurzlebige JS-Objekte,
   * und `indexOf` ueber einen langen String ist schneller als eine Schleife
   * ueber ein Array.
   */
  searchIndex() {
    if (this._searchIndex) return this._searchIndex;
    if (!this.text) return null;

    const titles = [];
    const starts = new Uint32Array(this.n);
    let blob = '';
    for (let index = 0; index < this.n; index++) {
      const slice = this.text.subarray(this.offsets[index], this.offsets[index + 1]);
      const title = decoder.decode(slice).split(this.separator)[0] ?? '';
      starts[index] = blob.length;
      blob += title.toLowerCase() + '\n';
      titles.push(title);
    }
    this._titles = titles;
    this._searchIndex = { blob, starts };
    return this._searchIndex;
  }

  /** Indizes aller Punkte, deren Titel `query` enthaelt. */
  search(query) {
    const index = this.searchIndex();
    if (!index || !query) return [];
    const needle = query.toLowerCase();
    const hits = [];
    let position = index.blob.indexOf(needle);
    while (position !== -1) {
      // Von der Fundstelle im Blob zurueck auf den Punktindex.
      let low = 0;
      let high = this.n - 1;
      while (low < high) {
        const middle = (low + high + 1) >> 1;
        if (index.starts[middle] <= position) low = middle;
        else high = middle - 1;
      }
      if (hits[hits.length - 1] !== low) hits.push(low);
      position = index.blob.indexOf(needle, position + 1);
    }
    return hits;
  }

  title(index) {
    this.searchIndex();
    return this._titles ? this._titles[index] : '';
  }
}
