import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const viewerDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(viewerDir, '..');
const dataRoot = path.join(repoRoot, 'data');

/**
 * Liefert `data/` unter `/data/` aus, ohne es in den Viewer zu kopieren.
 *
 * Die beiden Haelften des Projekts kommunizieren ueber Dateien (siehe
 * docs/decisions.md E7). Das Artefaktverzeichnis liegt deshalb bewusst
 * ausserhalb von `viewer/` — es gehoert der Pipeline, nicht dem Frontend.
 * Ein Symlink oder eine Kopie waere die Alternative; beide gehen beim naechsten
 * Pipeline-Lauf schief oder veralten still.
 */
function serveArtifacts() {
  return {
    name: 'lse-serve-artifacts',

    configureServer(server) {
      server.middlewares.use('/data', (req, res, next) => {
        // Nur den Pfad, ohne Query; und aufgeloest, damit `..` nicht aus
        // dataRoot herausfuehren kann.
        const rel = decodeURIComponent((req.url ?? '/').split('?')[0]);
        const target = path.resolve(dataRoot, '.' + rel);
        if (target !== dataRoot && !target.startsWith(dataRoot + path.sep)) {
          res.statusCode = 403;
          res.end('forbidden');
          return;
        }
        fs.stat(target, (err, stat) => {
          if (err || !stat.isFile()) {
            // Bewusst 404 statt `next()`: Vites SPA-Fallback wuerde sonst
            // `index.html` mit Status 200 ausliefern, und ein fehlendes
            // Artefakt kaeme als HTML im Binaerparser an — ein Fehler, der
            // sich als kaputtes Datenformat tarnt statt als fehlende Datei.
            res.statusCode = 404;
            res.end('not found');
            return;
          }
          res.setHeader(
            'Content-Type',
            target.endsWith('.json') ? 'application/json' : 'application/octet-stream',
          );
          res.setHeader('Content-Length', stat.size);
          fs.createReadStream(target).pipe(res);
        });
      });
    },

    // Fuer den Build wandert nur das eingecheckte Sample mit. Vollstaendige
    // Laeufe sind zu gross fuers Repo und reisen ueber Release-Assets.
    closeBundle() {
      const from = path.join(dataRoot, 'sample');
      if (!fs.existsSync(from)) return;
      const to = path.join(viewerDir, 'dist', 'data', 'sample');
      fs.mkdirSync(to, { recursive: true });
      for (const name of fs.readdirSync(from)) {
        fs.copyFileSync(path.join(from, name), path.join(to, name));
      }
    },
  };
}

export default defineConfig({
  plugins: [serveArtifacts()],
  server: {
    fs: { allow: [viewerDir, dataRoot] },
  },
  build: {
    target: 'esnext', // WebGPU-Pfad nutzt top-level await
  },
});
