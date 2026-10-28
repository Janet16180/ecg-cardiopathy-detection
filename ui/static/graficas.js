// Gráficas en SVG sin librerías externas.
// Cada función recibe un contenedor vacío, un DataFrame en formato "split" ({index, columns, data})
// tal como lo manda la API, y las opciones del bloque.

const NS = "http://www.w3.org/2000/svg";
const FUENTE = 'system-ui, -apple-system, "Segoe UI", sans-serif';
const lienzo = document.createElement("canvas").getContext("2d");
// Idioma de los números y de las pocas etiquetas que escriben las gráficas; lo fija app.js
let LOCALE = "es-MX";
let T = { sinDato: "sin dato", mediana: "mediana", cuartiles: "Q1 a Q3", bigotes: "bigotes", a: "a", ecgs: "ECGs", ecgAria: "ECG de 12 derivaciones" };
let compactador = new Intl.NumberFormat(LOCALE, { notation: "compact", maximumFractionDigits: 1 });

export function fijarIdioma(locale, textos) {
  LOCALE = locale;
  T = textos;
  compactador = new Intl.NumberFormat(locale, { notation: "compact", maximumFractionDigits: 1 });
}
const tip = document.getElementById("tooltip");

export function color(nombre) {
  return getComputedStyle(document.documentElement).getPropertyValue(`--${nombre}`).trim();
}

function nodo(tag, attrs = {}, padre = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (padre) padre.appendChild(e);
  return e;
}

function texto(padre, x, y, contenido, clase, attrs = {}) {
  const t = nodo("text", { x, y, class: clase, ...attrs }, padre);
  t.textContent = contenido;
  return t;
}

function medir(t, tam = 12) {
  lienzo.font = `${tam}px ${FUENTE}`;
  return lienzo.measureText(t).width;
}

// Recorta un texto a un ancho máximo; el texto completo queda en un <title>
function etiqueta(padre, x, y, contenido, max, attrs = {}) {
  let corto = contenido;
  while (corto.length > 1 && medir(corto + (corto === contenido ? "" : "…")) > max) corto = corto.slice(0, -1);
  const t = texto(padre, x, y, corto === contenido ? contenido : corto + "…", "t-etiqueta", attrs);
  if (corto !== contenido) nodo("title", {}, t).textContent = contenido;
  return t;
}

export function formatear(v, formato) {
  if (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))) return "—";
  if (typeof v !== "number") return String(v);
  if (formato === "pct") return `${v.toLocaleString(LOCALE, { useGrouping: "always", maximumFractionDigits: 1 })}%`;
  if (formato === "entero") return Math.round(v).toLocaleString(LOCALE, { useGrouping: "always" });
  if (typeof formato === "number") return v.toLocaleString(LOCALE, { useGrouping: "always", minimumFractionDigits: formato, maximumFractionDigits: formato });
  return v.toLocaleString(LOCALE, { useGrouping: "always", maximumFractionDigits: 3 });
}

function etiquetaEje(v, formato) {
  if (formato === "pct") return `${v}%`;
  return Math.abs(v) >= 1000 ? compactador.format(v) : v.toLocaleString(LOCALE, { maximumFractionDigits: 2 });
}

function rgb(hex) {
  return [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
}

function tintaSobre(hex) {
  const [r, g, b] = rgb(hex).map(c => c / 255);
  return 0.299 * r + 0.587 * g + 0.114 * b > 0.55 ? "#0b0b0b" : "#ffffff";
}

function pasoBonito(rango, n) {
  const paso0 = rango / n;
  const mag = 10 ** Math.floor(Math.log10(paso0));
  return [1, 2, 2.5, 5, 10].map(m => m * mag).find(p => p >= paso0);
}

// Marcas de eje de lo a hi con pasos redondos
function marcas(hi, n = 5, lo = 0) {
  if (!(hi > lo)) return [lo, lo + 1];
  const paso = pasoBonito(hi - lo, n);
  const salida = [];
  for (let v = Math.floor(lo / paso) * paso; ; v += paso) {
    salida.push(+v.toFixed(10));
    if (v >= hi - 1e-9) break;
  }
  return salida;
}

function barraH(x0, x1, y, h, r = 4) {
  const w = Math.max(x1 - x0, 0);
  r = Math.min(r, w, h / 2);
  return `M${x0},${y}H${x1 - r}A${r},${r} 0 0 1 ${x0 + w},${y + r}V${y + h - r}A${r},${r} 0 0 1 ${x1 - r},${y + h}H${x0}Z`;
}

function barraV(x, base, tope, w, r = 4) {
  const h = Math.max(base - tope, 0);
  r = Math.min(r, h, w / 2);
  return `M${x},${base}V${tope + r}A${r},${r} 0 0 1 ${x + r},${tope}H${x + w - r}A${r},${r} 0 0 1 ${x + w},${tope + r}V${base}Z`;
}

function leyenda(cont, series, colores) {
  const div = document.createElement("div");
  div.className = "leyenda";
  series.forEach((s, i) => {
    const span = document.createElement("span");
    const muestra = document.createElement("i");
    muestra.style.background = colores[i];
    span.append(muestra, document.createTextNode(s));
    div.append(span);
  });
  cont.append(div);
}

// Tooltip: el valor primero, la etiqueta después; también con el teclado
export function conTooltip(elemento, contenido, enfocable = true) {
  if (enfocable) elemento.setAttribute("tabindex", "0");
  elemento.classList.add("marca-dato");
  const mostrar = (x, y) => {
    const { titulo, filas } = contenido();
    tip.replaceChildren();
    const cabeza = document.createElement("div");
    cabeza.className = "tip-titulo";
    cabeza.textContent = titulo;
    tip.append(cabeza);
    for (const f of filas) {
      const fila = document.createElement("div");
      fila.className = "tip-fila";
      if (f.color) {
        const clave = document.createElement("i");
        clave.className = "tip-clave";
        clave.style.background = f.color;
        fila.append(clave);
      }
      const valor = document.createElement("b");
      valor.textContent = f.valor;
      fila.append(valor);
      if (f.etiqueta) {
        const nombre = document.createElement("span");
        nombre.textContent = f.etiqueta;
        fila.append(nombre);
      }
      tip.append(fila);
    }
    tip.style.display = "block";
    const r = tip.getBoundingClientRect();
    tip.style.left = `${Math.max(8, Math.min(x + 14, innerWidth - r.width - 8))}px`;
    tip.style.top = `${y + 14 + r.height > innerHeight ? y - r.height - 10 : y + 14}px`;
  };
  elemento.addEventListener("pointermove", e => mostrar(e.clientX, e.clientY));
  elemento.addEventListener("pointerleave", ocultarTooltip);
  elemento.addEventListener("focus", () => {
    const r = elemento.getBoundingClientRect();
    mostrar(r.right, r.top);
  });
  elemento.addEventListener("blur", ocultarTooltip);
}

export function ocultarTooltip() {
  tip.style.display = "none";
}

// Barras horizontales, una serie
export function barras(cont, df, op = {}) {
  const cats = df.index.map(String);
  const vals = df.data.map(f => f[0]);
  const fmt = v => formatear(v, op.formato);
  const W = Math.max(cont.clientWidth, 280);
  const izq = Math.min(Math.max(...cats.map(c => medir(c))) + 14, W * 0.38);
  const der = Math.max(...vals.map(v => medir(fmt(v), 11.5))) + 14;
  const banda = 30, grueso = 18, arriba = 6, abajo = 38;
  const H = arriba + banda * cats.length + abajo;
  const t = marcas(Math.max(...vals.filter(Number.isFinite), 0));
  const x = v => izq + (W - izq - der) * v / t.at(-1);
  const s = nodo("svg", { width: W, height: H, role: "img" }, cont);
  for (const v of t) {
    nodo("line", { x1: x(v), x2: x(v), y1: arriba, y2: H - abajo, class: "rejilla" }, s);
    texto(s, x(v), H - abajo + 16, etiquetaEje(v, op.formato), "t-eje", { "text-anchor": "middle" });
  }
  texto(s, izq + (W - izq - der) / 2, H - 4, op.eje ?? "", "t-eje", { "text-anchor": "middle" });
  nodo("line", { x1: izq, x2: izq, y1: arriba, y2: H - abajo, class: "base" }, s);
  const c1 = color("s1");
  cats.forEach((c, i) => {
    const y = arriba + i * banda + (banda - grueso) / 2;
    etiqueta(s, izq - 8, y + grueso / 2 + 4, c, izq - 14, { "text-anchor": "end" });
    if (!Number.isFinite(vals[i])) {
      texto(s, izq + 6, y + grueso / 2 + 4, T.sinDato, "t-valor");
      return;
    }
    const g = nodo("g", {}, s);
    nodo("rect", { x: izq, y: arriba + i * banda, width: W - izq, height: banda, fill: "transparent" }, g);
    nodo("path", { d: barraH(izq, x(vals[i]), y, grueso), fill: c1 }, g);
    texto(g, x(vals[i]) + 6, y + grueso / 2 + 4, fmt(vals[i]), "t-valor");
    conTooltip(g, () => ({ titulo: c, filas: [{ valor: fmt(vals[i]), etiqueta: op.eje, color: c1 }] }));
  });
}

// Barras al 100%, una por fila, con leyenda arriba
export function apiladas(cont, df, op = {}) {
  const filas = df.index.map(String);
  const series = df.columns.map(String);
  const colores = op.colores.map(color);
  leyenda(cont, series, colores);
  const W = Math.max(cont.clientWidth, 280);
  const izq = Math.min(Math.max(...filas.map(f => medir(f))) + 14, W * 0.38);
  const der = 22, banda = 32, grueso = 20, arriba = 4, abajo = 26;
  const H = arriba + banda * filas.length + abajo;
  const x = v => izq + (W - izq - der) * v / 100;
  const s = nodo("svg", { width: W, height: H, role: "img" }, cont);
  for (const v of [0, 25, 50, 75, 100]) {
    nodo("line", { x1: x(v), x2: x(v), y1: arriba, y2: H - abajo, class: "rejilla" }, s);
    texto(s, x(v), H - abajo + 16, `${v}%`, "t-eje", { "text-anchor": "middle" });
  }
  filas.forEach((f, i) => {
    const y = arriba + i * banda + (banda - grueso) / 2;
    etiqueta(s, izq - 8, y + grueso / 2 + 4, f, izq - 14, { "text-anchor": "end" });
    const g = nodo("g", {}, s);
    nodo("rect", { x: izq, y: arriba + i * banda, width: W - izq, height: banda, fill: "transparent" }, g);
    const valores = df.data[i].map(v => v ?? 0);
    const ultimo = valores.findLastIndex(v => v > 0);
    let acumulado = 0;
    valores.forEach((v, j) => {
      if (v <= 0) return;
      const x0 = x(acumulado);
      const x1 = x(Math.min(acumulado + v, 100)) - (j < ultimo ? 2 : 0);   // 2 px de separación entre tramos
      acumulado += v;
      nodo("path", { d: j === ultimo ? barraH(x0, x1, y, grueso) : `M${x0},${y}H${x1}V${y + grueso}H${x0}Z`, fill: colores[j] }, g);
      const rotulo = `${Math.round(v)}%`;
      if (x1 - x0 >= medir(rotulo, 11) + 10) {
        texto(g, (x0 + x1) / 2, y + grueso / 2 + 4, rotulo, "t-valor", { "text-anchor": "middle", "font-size": 11, style: `fill:${tintaSobre(colores[j])}` });
      }
    });
    conTooltip(g, () => ({ titulo: f, filas: series.map((nombre, j) => ({ valor: formatear(df.data[i][j], "pct"), etiqueta: nombre, color: colores[j] })) }));
  });
}

function interpolar(paradas, t) {
  t = Math.max(0, Math.min(1, t));
  const p = t * (paradas.length - 1);
  const i = Math.min(Math.floor(p), paradas.length - 2);
  const f = p - i;
  const c = paradas[i].map((a, k) => Math.round(a + (paradas[i + 1][k] - a) * f));
  return "#" + c.map(v => v.toString(16).padStart(2, "0")).join("");
}

// Mapa de calor con el valor en cada celda; NaN queda vacío
export function calor(cont, df, op = {}) {
  const filas = df.index.map(String);
  const cols = df.columns.map(String);
  const fmt = v => formatear(v, op.formato);
  const finitos = df.data.flat().filter(Number.isFinite);
  const max = op.maximo || Math.max(...finitos, 1);
  const min = op.minimo ?? 0;
  // op.intenso: el azul más fuerte es el valor más alto también en modo oscuro (métricas donde más es mejor)
  const paradas = (op.intenso ? ["i0", "i1", "i2", "i3", "i4", "i5"] : ["c0", "c1", "c2", "c3", "c4", "c5"]).map(n => rgb(color(n)));
  const W0 = Math.max(cont.clientWidth, 280);
  const izq = Math.min(Math.max(...filas.map(f => medir(f))) + 14, 300);
  const minimo = Math.max(46, ...finitos.map(v => medir(fmt(v), 11) + 14));
  const celda = Math.max(minimo, Math.min(130, (W0 - izq - 8) / cols.length));
  const rotar = cols.some(c => medir(c, 11.5) > celda - 8);
  const largo = Math.min(Math.max(...cols.map(c => medir(c, 11.5))), 190);
  const arriba = rotar ? largo * 0.71 + 18 : 24;
  const alto = 28;
  const W = Math.max(W0, izq + celda * cols.length + (rotar ? largo * 0.71 : 8));
  const H = arriba + alto * filas.length + 4;
  const s = nodo("svg", { width: W, height: H, role: "img" }, cont);
  cols.forEach((c, j) => {
    const cx = izq + celda * j + celda / 2;
    if (rotar) etiqueta(s, cx, arriba - 8, c, 190, { transform: `rotate(-45 ${cx} ${arriba - 8})`, "font-size": 11.5 });
    else texto(s, cx, arriba - 8, c, "t-etiqueta", { "text-anchor": "middle", "font-size": 11.5 });
  });
  filas.forEach((f, i) => {
    const y = arriba + i * alto;
    etiqueta(s, izq - 8, y + alto / 2 + 4, f, izq - 14, { "text-anchor": "end" });
    df.data[i].forEach((v, j) => {
      if (!Number.isFinite(v)) return;
      const fondo = interpolar(paradas, (v - min) / (max - min));
      const g = nodo("g", {}, s);
      nodo("rect", { x: izq + celda * j + 1, y: y + 1, width: celda - 2, height: alto - 2, rx: 3, fill: fondo }, g);
      texto(g, izq + celda * j + celda / 2, y + alto / 2 + 4, fmt(v), "t-valor", { "text-anchor": "middle", "font-size": 11, style: `fill:${tintaSobre(fondo)}` });
      conTooltip(g, () => ({ titulo: `${f} · ${cols[j]}`, filas: [{ valor: fmt(v) }] }), false);
    });
  });
  const escala = document.createElement("div");
  escala.className = "escala";
  const barra = document.createElement("i");
  barra.style.background = `linear-gradient(to right, ${paradas.map(c => `rgb(${c.join(",")})`).join(", ")})`;
  escala.append(document.createTextNode(fmt(min)), barra, document.createTextNode(fmt(max)));
  cont.append(escala);
}

// Histogramas en paneles pequeños, uno por columna, cada uno con su propia escala vertical
export function histogramas(cont, df, op = {}) {
  const bordes = df.index.map(Number);
  const paneles = df.columns.map(String);
  const colores = op.colores ? op.colores.map(color) : paneles.map(() => color("s1"));
  const ancho = op.ancho_bin ?? bordes[1] - bordes[0];
  const W = Math.max(cont.clientWidth, 280);
  const porFila = Math.max(1, Math.min(paneles.length, Math.floor(W / 250)));
  const pw = W / porFila, ph = 176;
  const H = Math.ceil(paneles.length / porFila) * ph + 22;
  const xmin = bordes[0], xmax = bordes.at(-1) + ancho;
  const marcasX = marcas(xmax, 5, xmin).filter(v => v <= xmax);
  const s = nodo("svg", { width: W, height: H, role: "img" }, cont);
  paneles.forEach((panel, k) => {
    const ox = (k % porFila) * pw, oy = Math.floor(k / porFila) * ph;
    const izq = ox + 44, der = ox + pw - 14, arriba = oy + 28, abajo = oy + ph - 30;
    const conteos = df.data.map(f => f[k] ?? 0);
    const t = marcas(Math.max(...conteos), 3);
    const y = v => abajo - (abajo - arriba) * v / t.at(-1);
    const x = v => izq + (der - izq) * (v - xmin) / (xmax - xmin);
    texto(s, izq, oy + 14, panel, "t-panel");
    for (const v of t) {
      nodo("line", { x1: izq, x2: der, y1: y(v), y2: y(v), class: "rejilla" }, s);
      texto(s, izq - 6, y(v) + 4, etiquetaEje(v), "t-eje", { "text-anchor": "end" });
    }
    for (const v of marcasX) texto(s, x(v), abajo + 16, etiquetaEje(v), "t-eje", { "text-anchor": "middle" });
    nodo("line", { x1: izq, x2: der, y1: abajo, y2: abajo, class: "base" }, s);
    bordes.forEach((b, i) => {
      if (conteos[i] <= 0) return;
      const x0 = x(b) + 1, x1 = x(b + ancho) - 1;
      const g = nodo("g", {}, s);
      nodo("rect", { x: x(b), y: arriba, width: x1 - x0 + 2, height: abajo - arriba, fill: "transparent" }, g);
      nodo("path", { d: barraV(x0, abajo, y(conteos[i]), x1 - x0, 2), fill: colores[k] }, g);
      conTooltip(g, () => ({ titulo: `${panel} · ${b}–${b + ancho}`,
                            filas: [{ valor: formatear(conteos[i], "entero"), etiqueta: op.unidad, color: colores[k] }] }), false);
    });
  });
  texto(s, W / 2, H - 4, op.eje ?? "", "t-eje", { "text-anchor": "middle" });
}

// Diagramas de caja horizontales: cuartiles y bigotes calculados en el servidor
export function cajas(cont, df, op = {}) {
  const filas = df.index.map(String);
  const [ib, iq1, im, iq3, is, ine] = [0, 1, 2, 3, 4, 5];   // bigote inferior, Q1, mediana, Q3, bigote superior, ECGs
  const W = Math.max(cont.clientWidth, 260);
  const izq = Math.min(Math.max(...filas.map(f => medir(f))) + 14, W * 0.38), der = 16;
  const banda = 30, grueso = 16, arriba = 4, abajo = 38;
  const H = arriba + banda * filas.length + abajo;
  const t = marcas(Math.max(...df.data.map(f => f[is])), 4, Math.min(...df.data.map(f => f[ib])));
  const x = v => izq + (W - izq - der) * (v - t[0]) / (t.at(-1) - t[0]);
  const s = nodo("svg", { width: W, height: H, role: "img" }, cont);
  for (const v of t) {
    nodo("line", { x1: x(v), x2: x(v), y1: arriba, y2: H - abajo, class: "rejilla" }, s);
    texto(s, x(v), H - abajo + 16, etiquetaEje(v), "t-eje", { "text-anchor": "middle" });
  }
  texto(s, izq + (W - izq - der) / 2, H - 4, op.eje ?? "", "t-eje", { "text-anchor": "middle" });
  const [cCaja, cMediana, cBigote] = [color("caja"), color("s1"), color("tinta-3")];
  filas.forEach((f, i) => {
    const d = df.data[i];
    const cy = arriba + i * banda + banda / 2;
    etiqueta(s, izq - 8, cy + 4, f, izq - 14, { "text-anchor": "end" });
    const g = nodo("g", {}, s);
    nodo("rect", { x: izq, y: arriba + i * banda, width: W - izq, height: banda, fill: "transparent" }, g);
    nodo("line", { x1: x(d[ib]), x2: x(d[is]), y1: cy, y2: cy, stroke: cBigote, "stroke-width": 1.5 }, g);
    for (const v of [d[ib], d[is]]) nodo("line", { x1: x(v), x2: x(v), y1: cy - 5, y2: cy + 5, stroke: cBigote, "stroke-width": 1.5 }, g);
    nodo("rect", { x: x(d[iq1]), y: cy - grueso / 2, width: Math.max(x(d[iq3]) - x(d[iq1]), 1), height: grueso, rx: 3, fill: cCaja }, g);
    nodo("line", { x1: x(d[im]), x2: x(d[im]), y1: cy - grueso / 2, y2: cy + grueso / 2, stroke: cMediana, "stroke-width": 2.5 }, g);
    conTooltip(g, () => ({ titulo: `${f} (${formatear(d[ine], "entero")} ${T.ecgs})`, filas: [
      { valor: formatear(d[im]), etiqueta: T.mediana },
      { valor: `${formatear(d[iq1])} ${T.a} ${formatear(d[iq3])}`, etiqueta: T.cuartiles },
      { valor: `${formatear(d[ib])} ${T.a} ${formatear(d[is])}`, etiqueta: T.bigotes },
    ] }));
  });
}

// Columnas agrupadas: una categoría por grupo y una serie por color
export function agrupadas(cont, df, op = {}) {
  const cats = df.index.map(String);
  const series = df.columns.map(String);
  const colores = op.colores.map(color);
  leyenda(cont, series, colores);
  const W = Math.max(cont.clientWidth, 280), H = 260;
  const izq = 44, der = 8, arriba = 10, abajo = 28;
  const t = marcas(Math.max(...df.data.flat().filter(Number.isFinite)), 4);
  const y = v => (H - abajo) - (H - abajo - arriba) * v / t.at(-1);
  const grupo = (W - izq - der) / cats.length;
  const anchoBarra = Math.max(4, Math.min(24, (grupo * 0.72 - 2 * (series.length - 1)) / series.length));
  const anchoGrupo = anchoBarra * series.length + 2 * (series.length - 1);
  const s = nodo("svg", { width: W, height: H, role: "img" }, cont);
  for (const v of t) {
    nodo("line", { x1: izq, x2: W - der, y1: y(v), y2: y(v), class: "rejilla" }, s);
    texto(s, izq - 6, y(v) + 4, etiquetaEje(v), "t-eje", { "text-anchor": "end" });
  }
  nodo("line", { x1: izq, x2: W - der, y1: H - abajo, y2: H - abajo, class: "base" }, s);
  cats.forEach((c, i) => {
    const g = nodo("g", {}, s);
    nodo("rect", { x: izq + grupo * i, y: arriba, width: grupo, height: H - abajo - arriba, fill: "transparent" }, g);
    const x0 = izq + grupo * i + (grupo - anchoGrupo) / 2;
    series.forEach((_, j) => {
      const v = df.data[i][j];
      if (Number.isFinite(v)) nodo("path", { d: barraV(x0 + j * (anchoBarra + 2), H - abajo, y(v), anchoBarra), fill: colores[j] }, g);
    });
    texto(s, izq + grupo * i + grupo / 2, H - abajo + 16, c, "t-etiqueta", { "text-anchor": "middle" });
    conTooltip(g, () => ({ titulo: c, filas: series.map((nombre, j) => ({ valor: `${formatear(df.data[i][j], op.formato)} ${op.eje ?? ""}`, etiqueta: nombre, color: colores[j] })) }));
  });
}

// Líneas con marcadores, una por columna; el eje vertical se ajusta a los datos (curvas de aprendizaje)
export function lineas(cont, df, op = {}) {
  const cats = df.index.map(String);
  const series = df.columns.map(String);
  const colores = op.colores.map(color);
  const fmt = v => formatear(v, op.formato);
  leyenda(cont, series, colores);
  const W = Math.max(cont.clientWidth, 280), H = 300;
  const izq = 58, der = 16, arriba = 12, abajo = 48;
  const finitos = df.data.flat().filter(Number.isFinite);
  const t = marcas(Math.max(...finitos), 5, Math.min(...finitos));
  const y = v => (H - abajo) - (H - abajo - arriba) * (v - t[0]) / (t.at(-1) - t[0]);
  const paso = (W - izq - der) / cats.length;
  const x = i => izq + paso * (i + 0.5);
  const s = nodo("svg", { width: W, height: H, role: "img" }, cont);
  for (const v of t) {
    nodo("line", { x1: izq, x2: W - der, y1: y(v), y2: y(v), class: "rejilla" }, s);
    texto(s, izq - 6, y(v) + 4, etiquetaEje(v), "t-eje", { "text-anchor": "end" });
  }
  const medio = arriba + (H - abajo - arriba) / 2;
  texto(s, 14, medio, op.eje ?? "", "t-eje", { "text-anchor": "middle", transform: `rotate(-90 14 ${medio})` });
  cats.forEach((c, i) => texto(s, x(i), H - abajo + 18, c, "t-etiqueta", { "text-anchor": "middle" }));
  texto(s, izq + (W - izq - der) / 2, H - 6, df.nombre_indice ?? "", "t-eje", { "text-anchor": "middle" });
  series.forEach((_, j) => {
    const puntos = cats.map((_, i) => [i, df.data[i][j]]).filter(([, v]) => Number.isFinite(v));
    nodo("path", { d: puntos.map(([i, v], k) => `${k ? "L" : "M"}${x(i)},${y(v)}`).join(""), fill: "none", stroke: colores[j], "stroke-width": 2 }, s);
    for (const [i, v] of puntos) nodo("circle", { cx: x(i), cy: y(v), r: 4.5, fill: colores[j], stroke: color("superficie"), "stroke-width": 2 }, s);
  });
  cats.forEach((c, i) => {
    const g = nodo("g", {}, s);
    nodo("rect", { x: izq + paso * i, y: arriba, width: paso, height: H - abajo - arriba, fill: "transparent" }, g);
    conTooltip(g, () => ({ titulo: c, filas: series.map((nombre, j) => ({ valor: fmt(df.data[i][j]), etiqueta: nombre, color: colores[j] })) }));
  });
}

// ECG de 12 derivaciones en papel milimetrado: 3 filas de 4 columnas (2.5 s cada una) y la II completa abajo
const DISPOSICION = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]];

// Áreas de interés: {inicio, fin} en segundos, derivaciones (null: las 12), color, punteada, titulo y filas del tooltip.
// En cada columna el tiempo t queda en x = t · 25 mm, así que basta recortar cada área a los 2.5 s de su columna.
function sombrear(s, areas, mm, alto) {
  for (const a of areas) {
    const g = nodo("g", {}, s);
    const c = color(a.color);
    const caja = (t0, t1, y) => nodo("rect", {
      x: t0 * 25 * mm, y: y + 1, width: Math.max((t1 - t0) * 25 * mm, 1), height: alto - 2, fill: c,
      "fill-opacity": a.punteada ? 0.12 : 0.26,
      ...(a.punteada ? { stroke: c, "stroke-width": 1.2, "stroke-dasharray": "4 3" } : {}),
    }, g);
    DISPOSICION.forEach((fila, r) => fila.forEach((nombre, col) => {
      if (a.derivaciones && !a.derivaciones.includes(nombre)) return;
      const t0 = Math.max(a.inicio, col * 2.5), t1 = Math.min(a.fin, (col + 1) * 2.5);
      if (t1 > t0) caja(t0, t1, r * alto);
    }));
    if (!a.derivaciones || a.derivaciones.includes("II")) caja(a.inicio, a.fin, 3 * alto);
    conTooltip(g, () => ({ titulo: a.titulo, filas: a.filas }), false);
  }
}

export function ecg(cont, d, areas = []) {
  const W = Math.max(cont.clientWidth, 640);
  const mm = W / 250;          // 10 s a 25 mm/s
  const alto = 30 * mm;        // 3 mV por fila a 10 mm/mV
  const H = alto * 4;
  const s = nodo("svg", { width: W, height: H, role: "img", "aria-label": T.ecgAria }, cont);

  let menor = "", mayor = "";
  for (let i = 0; i <= 250; i++) (i % 5 ? (menor += `M${i * mm},0V${H}`) : (mayor += `M${i * mm},0V${H}`));
  for (let i = 0; i <= 120; i++) (i % 5 ? (menor += `M0,${i * mm}H${W}`) : (mayor += `M0,${i * mm}H${W}`));
  if (mm >= 2.5) nodo("path", { d: menor, stroke: color("ecg-menor"), "stroke-width": 0.6 }, s);
  nodo("path", { d: mayor, stroke: color("ecg-mayor"), "stroke-width": 1 }, s);
  sombrear(s, areas, mm, alto);

  const fs = d.fs;
  const mediana = serie => {
    const v = serie.filter(Number.isFinite).sort((a, b) => a - b);
    return v.length ? v[Math.floor(v.length / 2)] : 0;
  };
  const base = d.senal.map(mediana);   // cada derivación se centra en su mediana
  const trazo = (k, i0, i1, x0, cy) => {
    let camino = "", pluma = false;
    for (let i = i0; i < Math.min(i1, d.senal[k].length); i++) {
      const v = d.senal[k][i];
      if (v === null) { pluma = false; continue; }
      camino += `${pluma ? "L" : "M"}${(x0 + (i - i0) / fs * 25 * mm).toFixed(1)},${(cy - (v - base[k]) * 10 * mm).toFixed(1)}`;
      pluma = true;
    }
    return camino;
  };
  const rotulo = (x, y, nombre) => texto(s, x, y, nombre, "t-panel", { style: `paint-order:stroke;stroke:${color("superficie")};stroke-width:3px` });

  let camino = "";
  const tramo = Math.round(2.5 * fs);
  DISPOSICION.forEach((fila, r) => fila.forEach((nombre, c) => {
    const cy = r * alto + alto / 2;
    camino += trazo(d.derivaciones.indexOf(nombre), c * tramo, (c + 1) * tramo, c * 62.5 * mm, cy);
    if (c) camino += `M${c * 62.5 * mm},${cy - 4 * mm}V${cy + 4 * mm}`;
    rotulo(c * 62.5 * mm + 6, r * alto + 16, nombre);
  }));
  camino += trazo(d.derivaciones.indexOf("II"), 0, 10 * fs, 0, 3 * alto + alto / 2);
  rotulo(6, 3 * alto + 16, "II");
  nodo("path", { d: camino, fill: "none", stroke: color("ecg-trazo"), "stroke-width": 1.2, "stroke-linejoin": "round" }, s);
}
