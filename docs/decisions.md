# Entscheidungen

Festgehaltene Projektentscheidungen mit Begründung. Grundlage ist
[`review-2026-08-03.md`](./review-2026-08-03.md). Wenn eine Entscheidung revidiert wird, wird sie hier
durchgestrichen und ersetzt, nicht gelöscht — der Grund, warum etwas *nicht* gemacht wurde, ist die
teuerste Information im Projekt.

---

## E1 — Korpus: Met Museum Open Access

**Entschieden:** ~490k gemeinfreie Werke des Metropolitan Museum of Art, gefiltert auf
`Is Public Domain = True` mit vorhandenem Bild. Zielgröße 100.000.

**Begründung.** Drei Kandidaten standen zur Wahl (arXiv-Abstracts, Met, eigene Daten). Ausschlaggebend
war nicht die Optik, sondern **CC0**: bei einem öffentlichen Projekt heißt das, die Thumbnails dürfen
selbst gehostet werden. Das ist der Unterschied zwischen "läuft unter einer URL" und einer
Rechtefrage. Dazu kommt:

- Die Cluster sind **ohne Klicken lesbar** — griechische Vasen, Ukiyo-e, Rüstungen sehen verschieden
  aus. Bei einem Textkorpus sind 100k Punkte grau und die Lesbarkeit muss komplett aus Farbe und
  Labels kommen.
- Es ist der einzige Kandidat, bei dem Phase 4c (Live-Prompt) auch das eindrucksvollste Feature ist:
  ein geteilter Bild-Text-Raum erlaubt "tippe eine Phrase, flieg zu den Bildern".
- Die Metadaten (Department, Culture, Period, Medium, Classification) sind reichhaltig genug, um
  Farbkanal, Facettenfilter **und** das Validierungslabel daraus zu ziehen.

**Preis, bewusst akzeptiert:** eine Download- und Thumbnail-Pipeline, die es bei arXiv nicht bräuchte.
Grob eine zusätzliche Sitzung.

**Ausstieg:** Falls das Go/No-Go-Tor in Phase 0b (siehe E6) rot ist, ist der Fallback
arXiv-Abstracts + EmbeddingGemma. Der Wechsel kostet dort einen Nachmittag; nach Phase 3 kostet er
einen Renderer-Neuschrieb.

---

## E2 — Zweck: öffentliches Portfolio-Stück

**Entschieden:** Statischer Viewer unter einer URL, Daten an GitHub-Release-Assets. Eine fremde Person
mit nur dem Link kann es benutzen.

**Konsequenzen, die daraus zwingend folgen:**

- Binäre, quantisierte Datenformate statt JSON. Nicht Optimierung, sondern Machbarkeit: 100k Punkte
  sind als JSON-Array-of-Objects ~18 MB mit 150–300 ms Main-Thread-Block, als int16-Binärdatei 600 KB
  mit null Parse-Zeit.
- Der Live-Prompt muss **ohne Server** funktionieren. Das erzwingt den kNN-Zentroid-Ansatz aus E4 —
  was ohnehin die bessere Technik ist.
- Es muss auf fremden Geräten laufen, nicht nur auf einer RTX 4060 Ti. Das ist der Grund für die
  Größen- und DPR-Clamps in E5.
- `data/sample/` mit 1.000 Punkten wird **committet**. Ein frischer Klon rendert mit `npm run dev` und
  null Python. Diese eine Entscheidung ist mehr wert als jedes README.

---

## E3 — Renderer: WebGPURenderer + TSL, instanzierte Quads

**Entschieden:** `three@0.185.1` exakt gepinnt, `WebGPURenderer` mit automatischem WebGL2-Fallback,
Punkte als instanzierte Quads.

**Warum nicht `WebGLRenderer`:** three.js' erklärte Position ist, dass `WebGLRenderer` keine großen
Features mehr bekommt. Ein neues Projekt Mitte 2026 dort zu starten heißt, auf dem Legacy-Pfad zu
starten. Der Fallback auf WebGL2 passiert aus einem einzigen Import, Abdeckung kostet also nichts.

**Warum nicht `gl.POINTS`** (vier unabhängige Gründe, alle in Phase 3 zu spät entdeckt):

1. `gl_PointSize` ist treiberseitig gedeckelt — `ALIASED_POINT_SIZE_RANGE` liefert **64 auf Apple
   Silicon**. Auf der 4060 Ti getunte Größen werden auf einem MacBook still geklemmt.
2. `gl.POINTS` clippt am Mittelpunkt, nicht an der Ausdehnung. Sprites verschwinden schlagartig am
   Bildrand.
3. **WebGPU hat keine Punktgröße.** `primitiveTopology: "point-list"` rendert 1×1 Pixel. Der Notausgang
   "später WebGPU" hätte exakt den Phase-3-Code entwertet.
4. Per-Punkt-Kontrolle (Rundung, dunkler Rand, Größen-Clamp) will man ohnehen.

Kosten: 6 Mio. statt 1 Mio. Vertices bei 1 Mio. Punkten ≈ 0,3 ms auf der Zielhardware. Draw Calls
bleiben 1.

**Gestrichen: "später WebGPU wenn es ruckelt".** Falsche Diagnose. Beide APIs landen auf denselben
ROPs; das erwartete Ruckeln ist füllratenlimitiert. 100k Punkte à 8px sind 5 MFrag — trivial. Aber in
einem Cluster erreichen die nächsten 2.000 Punkte je 200px: 63 MFrag, das Zwölffache der übrigen
98.000 zusammen. Das Ruckeln kommt von 2 % der Daten und wird von einer Zeile Größen-Clamp behoben,
nicht von einer neuen API.

---

## E4 — Live-Prompt: kNN-Zentroid statt `umap.transform()`

**Entschieden:** Prompt einbetten → Top-k nächste Nachbarn im **hochdimensionalen** Raum per Cosine →
Marker auf den distanzgewichteten Schwerpunkt von deren 3D-Positionen.

**Begründung.** Zwei unabhängige Reviews landeten aus verschiedenen Richtungen auf derselben Lösung.
Sie umgeht drei Probleme gleichzeitig:

- **Modality Gap.** CLIP-artige Modelle legen Bild- und Text-Embeddings in verschiedene Kegel des
  gemeinsamen Raums. Ein `transform()`ter Text-Prompt landet systematisch versetzt, möglicherweise
  außerhalb der Wolke. Bei einem Bildkorpus mit Textsuche ist das kein Randfall, sondern der
  Normalfall.
- **Modellgröße.** `umap.transform()` braucht `self._raw_data` und den pynndescent-Index
  (`umap_.py:3053`, `:3093`) — ein Pickle schleppt die komplette Trainingsmatrix mit. Bei 100k × 768
  fp32 sind das über 300 MB. Nicht ausliefer.bar an einen Browser.
- **Versionsfragilität.** UMAP-Pickles brechen über numpy-/numba-Versionswechsel. In sechs Monaten
  wäre das ein toter Feature-Pfad.

Dazu ist es in der UI ehrlicher erklärbar ("platziert zwischen seinen nächsten Nachbarn") als eine
undurchsichtige Projektion.

**Folgeentscheidung: kein densMAP.** `umap_.py:3087` wirft explizit `NotImplementedError` für
`transform()` unter densMAP. Auch wenn wir `transform()` nicht als Hauptpfad nutzen, halten wir ihn
als Fallback offen — und Dichteerhaltung ist in 3D ohnehin kaum wahrnehmbar, bei 2–3× Laufzeit.

**Späteres Upgrade, offengehalten:** ein destillierter MLP (PCA-50 → 3D) als ONNX, clientseitig. Dafür
muss Phase 2 den PCA-Zustand und die Embeddings persistieren — passiert ohnehin.

---

## E5 — Navigation: Orbit um bewegliches Ziel, Free-Fly als begrenzter Toggle

**Entschieden:** Default ist gedämpfter Orbit um ein bewegliches Ziel, initial auf dem
Wolkenschwerpunkt. Doppelklick auf einen Punkt tweent das Ziel dorthin (500–700 ms, Ease-in-out).
Free-Fly bleibt als Toggle, aber auf eine Kugel von ~2,5× Wolkenradius geclampt. Welt-Up gesperrt,
kein Roll.

**Das widerspricht bewusst dem ursprünglichen Pitch "begehbarer Raum".** Ein UMAP-Ergebnis ist ein
begrenztes, klumpiges Objekt ohne Boden, Horizont, Schwerkraft und Maßstab. Freies 6-DOF-Fliegen darin
heißt: überschießen, im featurelosen Schwarz landen, nicht zurückfinden. Die Wolke ist ein *Objekt*,
kein *Ort*.

Das Gefühl von Bewegung durch den Raum bleibt erhalten — der Doppelklick-Flug ist die
Schlüsselinteraktion des ganzen Projekts, weil er Navigation, Exploration und "wie komme ich zu dem
Cluster da drüben" auf einmal löst, und weil Verirren unmöglich ist, solange das Ziel immer ein echter
Punkt ist.

**Überprüfung in Phase 0b:** beide Schemata werden bei N=1.000 gebaut und selbst durchgeflogen. Kostet
dort zwei Stunden. Falls Free-Fly sich wider Erwarten besser anfühlt, wird diese Entscheidung
revidiert — mit Begründung, hier.

**Kein `logarithmicDepthBuffer`.** Near/Far werden aus den echten Wolkengrenzen gesetzt
(`near = 0.005 × radius`, `far = 8 × radius`, Verhältnis ~1000). Damit tritt Z-Fighting bei einem
24-Bit-Tiefenpuffer nicht auf. `logarithmicDepthBuffer` schreibt `gl_FragDepth`, deaktiviert Early-Z
und greift genau den Flaschenhals aus E3 an.

---

## E6 — Phase 0b ist ein Go/No-Go-Tor

**Entschieden:** Vor der Skalierung auf 100k wird bei N=1.000 geprüft, ob der Raum überhaupt etwas
bedeutet. Das Tor besteht aus zwei Teilen:

1. **Shuffle-Control.** Dieselbe Pipeline auf zufällig gemischten Embeddings laufen lassen. Was dort an
   Struktur erscheint, ist die UMAP-Artefakt-Baseline. **UMAP erzeugt aus reinem Gauß-Rauschen
   überzeugend aussehende Cluster** — ein reiner Sichtcheck kann also nicht durchfallen und ist als
   Kontrolle wertlos.
2. **kNN-Overlap als Zahl.** Top-k-Nachbarn hochdimensional vs. in 3D, mittlerer `|∩|/k` bei k=15.
   Von 768d auf 3d sind 0,10–0,25 normal; ~0,02 heißt Rauschen.

Wenn bekannte Labels sich nicht trennen: **Korpus wechseln, nicht Parameter tunen.** Das ist der
häufigste Todesgrund solcher Projekte, und in Phase 0b kostet die Korrektur einen Nachmittag.

---

## E7 — Repo-Form: Dateien als API

**Entschieden:** Ein Repo, zwei Hälften: `pipeline/` (Python, uv) und `viewer/` (Vite, three.js).

**Die zentrale architektonische Tatsache: die beiden Hälften kommunizieren ausschließlich über Dateien
auf der Platte.** Keine Imports, kein laufender Prozess zwischen ihnen. Das Artefaktverzeichnis mit
seinem `manifest.json` ist die eigentliche API des Projekts — und der Ort, an dem beide Seiten
gleichzeitig reviewt werden müssen.

**ID-Invariante, gilt projektweit und darf nie gebrochen werden:**
> Der Zeilenindex **ist** die ID. Korpus, Embeddings, Koordinaten, Farben, Nachbarn und Thumbnails
> werden in identischer Reihenfolge geschrieben.

**Datendateien:** `data/` ist gitignored, **kein git-lfs** (Quota-Ärger, macht das Repo klon-feindlich).
Regenerierung ist ein dokumentierter Befehl. Ausnahme: `data/sample/` mit 1.000 Punkten wird committet
(siehe E2).

---

## E9 — Facetten gemessen statt geraten; `medium` verworfen

**Entschieden:** Die Facetten sind `department`, `classification`, `culture`, `period`. Jede wird auf
16 Werte plus „Andere" gekappt.

**Grundlage** — Auszählung über alle 248.472 gemeinfreien Objekte:

| Feld | verschiedene Werte | Top-8 deckt |
|---|---|---|
| `department` | 19 | 83 % |
| `classification` | 693 | 60 % |
| `culture` | 5.242 | 71 % (davon 42 % „unbekannt") |
| `medium` | **37.876** | **26 %** |

`medium` war im ersten Entwurf als Facette vorgesehen und ist verworfen: Werte wie
„Stone, probably jet; incised" sind zu fein, um zu filtern. Es bleibt als Detailfeld erhalten.
`period` wird nicht aus dem gleichnamigen Freitextfeld gebildet, sondern aus den numerischen
Datumsfeldern zu Jahrhunderten gebündelt — die Textspalte reicht von „Edo period (1615–1868)" bis
leer, die Zahlenspalten sind fast durchgängig gesetzt.

---

## E10 — Klassenzahl vor dem Ziehen begrenzen, nicht danach gruppieren

**Entschieden:** Die Stichprobe wird auf die acht größten Labelklassen beschränkt, **bevor**
geschichtet gezogen wird. Klassen unter 20 Elementen fallen vorher weg.

**Der erste Entwurf machte es umgekehrt** — alle 19 Abteilungen ziehen, dann die größten sieben
einfärben und den Rest zu „Andere" zusammenfassen. Das Ergebnis war unbrauchbar: die geschichtete
Stichprobe macht alle Klassen ungefähr gleich groß, danach ist „die größten sieben" eine
Zufallsauswahl, und die übrigen zwölf landeten als **676 von 1.000 Punkten** in einem einzigen
grauen Sammeltopf. Zwei Drittel der Wolke in einer Farbe sind weder ein Farbkanal noch ein Testfall.

Vorher zu begrenzen löst beides auf einmal: das Label ist zugleich Farbkanal und Wahrheitsgrundlage
des Go/No-Go-Tors, mit acht ausgewogenen, visuell verschiedenen Klassen — exakt die Palettengröße.
Die acht größten Abteilungen decken 83 % der gemeinfreien Sammlung ab.

**Folgeregel: das Label wird nie gekappt.** Das Facetten-Kappen (E9) überspringt das Labelfeld. Es
auch dort anzuwenden hatte im ersten Entwurf die 19 Abteilungen auf 17 reduziert — und damit die
Wahrheitsgrundlage des Tors verändert, gegen die validiert wird.

---

## E11 — Der Shuffle-Control mischt Spalten, nicht Zeilen

**Entschieden:** Die Kontrolle permutiert jede Dimension unabhängig über alle Zeilen.

**Zeilen zu mischen wäre wirkungslos** — es ist dieselbe Punktmenge in anderer Reihenfolge, UMAP
fände exakt dieselbe Mannigfaltigkeit, und die „Kontrolle" sähe genauso gut aus wie die echten
Daten. Die Kontrolle muss die Korrelationen *zwischen* den Dimensionen auflösen. Eine unabhängige
Permutation je Spalte tut genau das: jede Dimension behält ihre Verteilung, aber kein Punkt ist mehr
eine sinnvolle Kombination.

Gemessen (synthetische Blobs, dim=32, k=10): Label-Reinheit 0,99 im Original, 0,99 nach
Zeilenpermutation, 0,31 nach Spaltenpermutation. Der Test hält das fest.

---

## E12 — Der Textblock ist zugleich der Suchindex, deshalb ohne URLs

**Entschieden:** `text.bin` trägt Datensätze getrennt durch ASCII 30 (Record Separator), Felder
getrennt durch ASCII 31. `link` steht **nicht** darin; der Viewer setzt die URL aus der Objekt-ID
zusammen.

**Warum ein Trenner, obwohl es schon eine Offsettabelle gibt.** Die Offsets sind **Byte**-Positionen,
JS-Strings zählen aber UTF-16-Einheiten — bei Umlauten, CJK-Zeichen oder Emoji laufen beide
auseinander. Ohne Trenner müsste der Browser jeden Datensatz einzeln dekodieren: bei 100.000 Punkten
100.000 `TextDecoder`-Aufrufe plus ebenso viele `split`-Aufrufe und kurzlebige Stringobjekte. Mit
Trenner wird **einmal** dekodiert, einmal gescannt, und die Suche ist danach ein `indexOf` über einen
zusammenhängenden String.

**Warum `link` raus muss.** Jede Met-URL enthält `/art/collection/search/`. Da der Block zugleich der
Suchindex ist, hätte eine Suche nach „art" oder „search" **jeden** Datensatz getroffen. Die URL folgt
einem festen Muster, ist also aus der Objekt-ID rekonstruierbar. Nebeneffekt: `text.bin` schrumpft
bei 100.000 Punkten von 16,67 auf 12,47 MB.

**Schutz vor stillen Fehlern:** ein Artefaktsatz ohne `record_separator` im Manifest lässt den Viewer
mit klarer Meldung abbrechen. Ohne diese Prüfung wären die Offsets um eins pro Datensatz verschoben
und das Detailpanel zeigte Bruchstücke fremder Einträge — der schlimmste Fehlermodus, weil er wie
Daten aussieht.

---

## E13 — `--unseeded` für Explorationsläufe

**Gemessen an 100.000 Punkten auf vier Kernen: 134 s mit Seed, 66 s ohne.**

`random_state` erzwingt `n_jobs=1` quer durch pynndescents kNN-Bau **und** die Layout-Epochen
(`umap_.py:2003`). Auf einer Maschine mit mehr Kernen ist der Faktor entsprechend größer.

Politik: unseeded explorieren, genau **ein** gesäter Lauf für das veröffentlichte Artefakt, Seed im
Manifest. Der Schalter macht das explizit statt es in der Config zu verstecken.

---

## E8 — Egress-Beschränkung der Entwicklungsumgebung

**Festgestellt am 2026-08-03, keine Entscheidung sondern eine Randbedingung.**

Die Remote-Session, in der entwickelt wird, blockt per Organisationsrichtlinie:

| Host | Status | Betroffen |
|---|---|---|
| `collectionapi.metmuseum.org` | ❌ 403 am Tunnel | Met-Objekt-API |
| `images.metmuseum.org` | ❌ 403 am Tunnel | Bild-Download, Thumbnails |
| `huggingface.co`, `cdn-lfs.huggingface.co` | ❌ 403 am Tunnel | alle Modellgewichte |
| `media.githubusercontent.com` | ✅ | **MetObjects.csv — die kompletten Metadaten** |
| PyPI, npm, GitHub | ✅ | Abhängigkeiten |

**Konsequenz für die Arbeitsteilung:**

- Hier **entwickelt und echt verifiziert**: Metadaten-Pipeline, Datenvertrag, PCA/UMAP/Validierung,
  kompletter Viewer, Tests.
- Hier **geschrieben, lokal auszuführen**: Bild-Download + Thumbnails, Modell-Embedding.

Deshalb ist `lse.synthetic` kein Spielzeug, sondern Infrastruktur: es erzeugt Gauß-Blobs mit Labels in
exakt den Zielformaten und entkoppelt den Viewer vollständig von den blockierten Schritten. Der
Renderer kann bei 1 Mio. Punkten belastet werden, bevor ein einziges echtes Embedding existiert.
