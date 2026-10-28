import * as g from "./graficas.js";
import { IDIOMAS, LOCALES, TEXTOS } from "./textos.js";

// Menú lateral: para agregar una página basta con una entrada aquí, su texto en textos.js y su función
const DATASETS = ["eda/resumen", "eda/pacientes", "eda/diagnosticos", "eda/clases", "eda/senales", "eda/duplicados", "visor"];
const MODELOS = ["experimentos/modelos", "experimentos/resultados", "experimentos/eleccion", "modelo"];
const MENU = [
  { accion: "registrar" },
  { ruta: "historial" },
  { grupo: "acerca", hijos: [
    { grupo: "datasets", hijos: DATASETS.map(ruta => ({ ruta })) },
    { grupo: "modelos", hijos: MODELOS.map(ruta => ({ ruta })) },
  ] },
];
const MIGAS = { historial: [], ...Object.fromEntries(DATASETS.map(ruta => [ruta, ["acerca", "datasets"]])),
                ...Object.fromEntries(MODELOS.map(ruta => [ruta, ["acerca", "modelos"]])) };
const DIBUJOS = { barras: g.barras, apiladas: g.apiladas, calor: g.calor, histogramas: g.histogramas, cajas: g.cajas, agrupadas: g.agrupadas, lineas: g.lineas };
const TEMAS = ["", "light", "dark"];

const estado = { originales: [], nuevo: "", seleccion: [], idioma: "es", pagina: 0, peticion: 0, dibujo: null, pendiente: null, ultimoId: null, fs: 500, modelo: null, areas: false };
const contenido = document.getElementById("contenido");
const todos = () => [...estado.originales, estado.nuevo];
const tr = () => TEXTOS[estado.idioma];
const nombre = d => tr().nombres[d] ?? d;

function h(tag, attrs = {}, ...hijos) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const hijo of hijos.flat(Infinity)) {
    if (hijo !== null && hijo !== undefined) e.append(hijo instanceof Node ? hijo : String(hijo));
  }
  return e;
}

function leer(clave) {
  try { return JSON.parse(localStorage.getItem(clave)); } catch { return null; }
}

function guardar(clave, valor) {
  try { localStorage.setItem(clave, JSON.stringify(valor)); } catch { /* sin almacenamiento, no pasa nada */ }
}

async function api(url, parametros, cuerpo = null) {
  parametros.set("lang", estado.idioma);
  const opciones = cuerpo ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cuerpo) } : {};
  const respuesta = await fetch(`${url}?${parametros}`, opciones);
  const datos = await respuesta.json();
  if (!respuesta.ok) throw new Error(typeof datos.detail === "string" ? datos.detail : tr().error(respuesta.status));
  return datos;
}

// La ruta, la selección y el idioma viajan en la URL (#/eda/clases?ds=Balanceado+v2&lang=it)
function escribirURL(ruta, conSeleccion) {
  const consulta = new URLSearchParams(conSeleccion ? estado.seleccion.map(d => ["ds", d]) : []);
  consulta.set("lang", estado.idioma);
  history.replaceState(null, "", `#/${ruta}?${consulta}`);
}

// Idioma, menú y barra lateral
function aplicarIdioma(idioma) {
  estado.idioma = idioma;
  guardar("idioma", idioma);
  document.documentElement.lang = idioma;
  document.title = tr().marca;
  g.fijarIdioma(LOCALES[idioma], tr().graficas);
  document.getElementById("marca").textContent = tr().marca;
  document.getElementById("boton-menu").textContent = tr().menu;
  document.getElementById("menu").setAttribute("aria-label", tr().navegacion);
  const idiomas = document.getElementById("idiomas");
  idiomas.setAttribute("aria-label", tr().idioma);
  for (const boton of idiomas.children) boton.setAttribute("aria-pressed", String(boton.dataset.idioma === idioma));
  rotularTema();
  construirMenu();
}

function construirIdiomas() {
  const idiomas = document.getElementById("idiomas");
  for (const idioma of IDIOMAS) {
    const boton = h("button", { type: "button", class: "idioma", "data-idioma": idioma, lang: idioma }, idioma.toUpperCase());
    boton.addEventListener("click", () => {
      if (idioma === estado.idioma) return;
      const [camino, consulta] = location.hash.replace(/^#\/?/, "").split("?");
      const parametros = new URLSearchParams(consulta ?? "");
      parametros.set("lang", idioma);
      history.replaceState(null, "", `#/${camino || "inicio"}?${parametros}`);
      aplicarIdioma(idioma);
      mostrar();
    });
    idiomas.append(boton);
  }
}

function nodoMenu(item, abiertos) {
  if (item.accion) {
    const boton = h("button", { type: "button", class: "boton boton-registrar" }, tr().registro.titulo);
    boton.addEventListener("click", () => document.getElementById("archivo-ecg").click());
    return [boton];
  }
  if (item.ruta) return [h("a", { class: "menu-enlace", href: `#/${item.ruta}`, "data-ruta": item.ruta }, tr().paginas[item.ruta][0])];
  const submenu = h("div", { class: "submenu" }, item.hijos.flatMap(hijo => nodoMenu(hijo, abiertos)));
  submenu.hidden = abiertos[item.grupo] === false;
  const boton = h("button", { type: "button", class: "menu-grupo", "aria-expanded": String(!submenu.hidden) }, tr().grupos[item.grupo]);
  boton.addEventListener("click", () => {
    submenu.hidden = !submenu.hidden;
    boton.setAttribute("aria-expanded", String(!submenu.hidden));
    abiertos[item.grupo] = !submenu.hidden;
    guardar("grupos", abiertos);
  });
  return [boton, submenu];
}

function construirMenu() {
  const abiertos = leer("grupos") ?? {};
  document.getElementById("menu").replaceChildren(...MENU.flatMap(item => nodoMenu(item, abiertos)));
}

function marcarMenu(ruta) {
  for (const enlace of document.querySelectorAll(".menu-enlace")) {
    if (enlace.dataset.ruta !== ruta) {
      enlace.removeAttribute("aria-current");
      continue;
    }
    enlace.setAttribute("aria-current", "page");
    for (let padre = enlace.parentElement; padre.classList.contains("submenu"); padre = padre.parentElement) {
      padre.hidden = false;   // abre los grupos que contienen la página actual
      padre.previousElementSibling.setAttribute("aria-expanded", "true");
    }
  }
  abrirMenuMovil(false);
}

function abrirMenuMovil(abrir) {
  document.getElementById("lateral").classList.toggle("abierta", abrir);
  document.getElementById("cortina").classList.toggle("abierta", abrir);
  document.getElementById("boton-menu").setAttribute("aria-expanded", String(abrir));
}

const temaActual = () => document.documentElement.dataset.theme ?? "";

function rotularTema() {
  document.getElementById("tema").textContent = tr().tema[temaActual()];
}

function iniciarTema() {
  document.getElementById("tema").addEventListener("click", () => {
    const siguiente = TEMAS[(TEMAS.indexOf(temaActual()) + 1) % TEMAS.length];
    if (siguiente) document.documentElement.dataset.theme = siguiente;
    else delete document.documentElement.dataset.theme;
    try { siguiente ? localStorage.setItem("tema", siguiente) : localStorage.removeItem("tema"); } catch { /* idem */ }
    rotularTema();
    redibujar();
  });
}

function mostrar() {
  const [camino, consulta] = location.hash.replace(/^#\/?/, "").split("?");
  const parametros = new URLSearchParams(consulta ?? "");
  const ds = parametros.getAll("ds");
  if (ds.some(d => todos().includes(d))) {
    estado.seleccion = todos().filter(d => ds.includes(d));
    guardar("seleccion", estado.seleccion);
  }
  const idioma = parametros.get("lang");
  if (IDIOMAS.includes(idioma) && idioma !== estado.idioma) aplicarIdioma(idioma);

  const analisis = camino.match(/^historial\/(\d+)$/);   // #/historial/12: un análisis guardado
  const ruta = analisis ? camino : camino in MIGAS ? camino : "inicio";
  estado.dibujo = null;
  estado.pagina++;     // invalida lo que pidió la página anterior
  estado.peticion++;
  g.ocultarTooltip();
  marcarMenu(analisis ? "historial" : ruta);
  window.scrollTo(0, 0);
  if (ruta.startsWith("eda/")) return paginaEda(ruta);
  escribirURL(ruta, false);
  if (analisis) return paginaAnalisis(Number(analisis[1]));
  if (ruta === "historial") return paginaHistorial();
  if (ruta === "visor") return paginaVisor();
  if (ruta === "modelo") return paginaModelo();
  if (ruta.startsWith("experimentos/")) return paginaExperimentos(ruta);
  return paginaInicio();
}

// Encabezado de cada página: ruta en el menú, título y descripción
function encabezado(ruta) {
  const [titulo, descripcion] = tr().paginas[ruta];
  const migas = [...MIGAS[ruta].map(grupo => tr().grupos[grupo]), titulo].join("  ›  ");
  return [MIGAS[ruta].length ? h("p", { class: "migas" }, migas) : null, h("h1", {}, titulo), h("p", { class: "bajada" }, descripcion)]
    .filter(Boolean);
}

// Bloques que manda la API
function tabla(d) {
  // Columnas de texto alineadas a la izquierda; las de números, a la derecha
  const esTexto = d.columns.map((_, j) => d.data.every(fila => typeof fila[j] === "string" || fila[j] === null));
  const clase = j => (esTexto[j] ? "texto" : "");
  return h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, d.nombre_indice ?? ""), d.columns.map((c, j) => h("th", { class: clase(j) }, String(c))))),
    h("tbody", {}, d.data.map((fila, i) => h("tr", {},
      h("td", {}, String(d.index[i])),
      fila.map((v, j) => h("td", { class: clase(j) }, g.formatear(v)))))));
}

function dibujarKpis(destino, kpis) {
  destino.replaceChildren(...kpis.map(k => h("div", { class: "kpi" },
    h("div", { class: "kpi-etiqueta" }, k.etiqueta),
    h("div", { class: "kpi-valor" }, g.formatear(k.valor, k.formato ?? "entero")),
    k.detalle ? h("div", { class: "kpi-detalle" }, k.detalle) : null)));
}

function dibujarBloques(destino, bloques) {
  destino.replaceChildren();
  for (const b of bloques) {
    if (b.tipo === "titulo") { destino.append(h("h2", { class: "subtitulo" }, b.titulo)); continue; }
    if (b.tipo === "aviso") { destino.append(h("p", { class: "aviso" }, b.nota)); continue; }
    if (b.tipo === "lista") {   // fortalezas y debilidades de un modelo
      destino.append(h("section", { class: "tarjeta completo" },
        h("div", { class: "tarjeta-cabeza" }, h("div", {}, h("h3", {}, b.titulo), b.nota ? h("p", { class: "nota" }, b.nota) : null)),
        h("div", { class: "listas" }, Object.entries(b.datos).map(([clave, items]) =>
          h("div", { class: `lista lista-${clave}` }, h("h4", {}, tr().listas[clave]), h("ul", {}, items.map(i => h("li", {}, i))))))));
      continue;
    }
    if (b.tipo === "kpis") {
      const kpis = h("div", { class: "kpis" });
      destino.append(kpis);
      dibujarKpis(kpis, b.datos);
      continue;
    }
    const cuerpo = h("div", { class: "cuerpo" });
    const esGrafica = b.tipo in DIBUJOS;
    let comoTabla = !esGrafica;
    const pintar = () => {
      cuerpo.replaceChildren();
      if (comoTabla) cuerpo.append(tabla(b.datos));
      else DIBUJOS[b.tipo](cuerpo, b.datos, b.opciones);
    };
    let boton = null;
    if (esGrafica) {
      boton = h("button", { type: "button", class: "vista" }, tr().verTabla);
      boton.addEventListener("click", () => {
        comoTabla = !comoTabla;
        boton.textContent = comoTabla ? tr().verGrafica : tr().verTabla;
        pintar();
      });
    }
    destino.append(h("section", { class: `tarjeta${b.opciones.ancho === "medio" ? "" : " completo"}` },
      h("div", { class: "tarjeta-cabeza" },
        h("div", {}, h("h3", {}, b.titulo), b.nota ? h("p", { class: "nota" }, b.nota) : null),
        boton),
      cuerpo));
    pintar();
  }
}

// Vuelve a dibujar la página actual (cambio de tamaño o de tema) sin pedir datos otra vez
function redibujar() {
  estado.dibujo?.();
}

function dibujarECG(papel, ecg, areas = []) {
  papel.replaceChildren();
  g.ecg(papel, ecg, areas);
}

// Inicio: solo el resultado del último ECG registrado
async function paginaInicio() {
  const resultado = h("div", { class: "registro-resultado" });
  contenido.replaceChildren(resultado);
  if (!estado.pendiente && !estado.ultimoId) return;
  const id = ++estado.peticion;
  resultado.replaceChildren(h("p", { class: "aviso" }, tr().registro.analizando));
  try {
    let r;
    if (estado.pendiente) {
      const pendiente = estado.pendiente;
      estado.pendiente = null;   // se envía una sola vez; después se lee del historial
      r = await api("/api/registrar", new URLSearchParams(), pendiente);
      estado.ultimoId = r.id;
    } else {
      r = await api(`/api/historial/${estado.ultimoId}`, new URLSearchParams());   // al volver o al cambiar de idioma
    }
    if (id !== estado.peticion) return;
    mostrarRegistro(resultado, r);
  } catch (error) {
    if (id !== estado.peticion) return;
    resultado.replaceChildren(h("p", { class: "aviso error" }, error.message));
  }
}

// Registrar nuevo ECG: botón de la barra lateral → elegir archivo → frecuencia de muestreo → procesar
function base64(archivo) {
  return new Promise((resolver, rechazar) => {
    const lector = new FileReader();
    lector.onload = () => resolver(lector.result.split(",")[1]);
    lector.onerror = () => rechazar(lector.error);
    lector.readAsDataURL(archivo);
  });
}

async function fsDeCabecera(archivos) {
  // En WFDB la frecuencia viene en la primera línea del .hea: "registro derivaciones fs muestras"
  const cabecera = archivos.find(a => a.name.toLowerCase().endsWith(".hea"));
  if (!cabecera) return null;
  const linea = (await cabecera.text()).split("\n").find(l => l.trim() && !l.startsWith("#"));
  const fs = parseFloat(linea?.trim().split(/\s+/)[2]);
  return Number.isFinite(fs) ? fs : null;
}

// El modelo elegido se guarda en el navegador; si ya no existe, se usa el predeterminado
function modeloValido(lista) {
  if (!lista.modelos.some(m => m.id === estado.modelo)) estado.modelo = lista.predeterminado;
}

async function pedirFrecuencia(archivos) {
  const t = tr().registro;
  const fsCabecera = await fsDeCabecera(archivos);
  const lista = await api("/api/modelos", new URLSearchParams());
  modeloValido(lista);
  const modelo = h("select", {}, lista.modelos.map(m => h("option", { value: m.id }, m.resumen)));
  modelo.value = estado.modelo;
  const fs = h("input", { type: "number", min: "50", max: "10000", step: "any", required: "",
                          value: String(fsCabecera ?? estado.fs) });
  fs.readOnly = fsCabecera !== null;
  const cancelar = h("button", { type: "button", class: "boton-secundario" }, t.cancelar);
  const formulario = h("form", { method: "dialog", class: "dialogo" },
    h("h2", {}, t.titulo),
    h("p", { class: "dialogo-archivos" }, archivos.map(a => a.name).join(", ")),
    h("label", { class: "campo" }, t.pideFs, fs),
    fsCabecera !== null ? h("p", { class: "nota" }, t.deCabecera) : null,
    lista.modelos.length ? h("label", { class: "campo" }, t.modelo, modelo) : null,
    h("p", { class: "nota" }, t.privacidad),
    h("div", { class: "dialogo-botones" }, cancelar, h("button", { type: "submit", class: "boton" }, t.confirmar)));
  const dialogo = document.getElementById("dialogo-registro");
  dialogo.replaceChildren(formulario);
  cancelar.addEventListener("click", () => dialogo.close());
  formulario.addEventListener("submit", async evento => {
    evento.preventDefault();
    dialogo.close();
    estado.fs = Number(fs.value);
    estado.modelo = modelo.value || estado.modelo;
    guardar("modelo", estado.modelo);
    estado.pendiente = {
      fs: estado.fs,
      modelo: estado.modelo,
      archivos: await Promise.all(archivos.map(async a => ({ nombre: a.name, contenido: await base64(a) }))),
    };
    if (location.hash.startsWith("#/inicio")) mostrar();
    else location.hash = "#/inicio";
  });
  dialogo.showModal();
  if (!fs.readOnly) fs.select();
}

function tarjetaModelo(m) {
  const t = tr().resultadoModelo;
  const pct = v => g.formatear(100 * v, "pct");
  const partes = [h("h3", {}, t.titulo)];
  if (m.motivo) {
    partes.push(h("p", {}, m.motivo));
  } else if (!m.conectado) {
    partes.push(h("p", {}, t.sinModelo));
  } else {
    partes.push(
      h("p", { class: `veredicto ${m.riesgo ? "con-riesgo" : "sin-riesgo"}` }, h("i", { "aria-hidden": "true" }), m.riesgo ? t.riesgo : t.sinRiesgo),
      h("div", { class: "probabilidad" }, h("span", { class: "probabilidad-valor" }, pct(m.probabilidad)), h("span", {}, t.probabilidad)),
      h("div", { class: "medidor", role: "img", "aria-label": `${pct(m.probabilidad)} · ${t.umbral(pct(m.umbral))}` },
        h("div", { class: "medidor-relleno", style: `width:${100 * m.probabilidad}%` }),
        h("div", { class: "medidor-umbral", style: `left:${100 * m.umbral}%` })),
      m.etiquetas ? detalleEtiquetas(m.etiquetas) : null);
  }
  return h("section", { class: `tarjeta resultado-modelo${m.conectado && !m.motivo ? "" : " sin-resultado"}` }, partes);
}

// Triángulo de alerta (SVG propio, sin librerías)
function iconoAlerta(titulo) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("class", "icono-alerta");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", titulo);
  svg.innerHTML = '<path d="M8 1.4c.4 0 .8.2 1 .6l6.3 11c.4.7-.1 1.6-.9 1.6H1.6c-.8 0-1.3-.9-.9-1.6L7 2c.2-.4.6-.6 1-.6z" fill="currentColor"/>'
    + '<path d="M8 5.6v4" stroke="#fff" stroke-width="1.7" stroke-linecap="round"/><circle cx="8" cy="12" r="1" fill="#fff"/>';
  const nombre = document.createElementNS("http://www.w3.org/2000/svg", "title");
  nombre.textContent = titulo;
  svg.prepend(nombre);
  return svg;
}

// Probabilidad de cada categoría y de los 5 diagnósticos más probables, con su punto de corte
function detalleEtiquetas(etiquetas) {
  const t = tr().resultadoModelo;
  const pct = v => g.formatear(100 * v, "pct");
  const fila = e => h("li", { class: `etiqueta-modelo${e.presente ? " presente" : ""}` },
    h("span", { class: "etiqueta-nombre" }, e.presente ? iconoAlerta(t.alerta) : h("i", { "aria-hidden": "true" }), e.etiqueta),
    h("span", { class: "etiqueta-valor" }, pct(e.probabilidad)),
    h("div", { class: "medidor", role: "img", "aria-label": `${e.etiqueta}: ${pct(e.probabilidad)} · ${t.umbral(pct(e.umbral))}` },
      h("div", { class: "medidor-relleno", style: `width:${100 * e.probabilidad}%` }),
      h("div", { class: "medidor-umbral", style: `left:${100 * e.umbral}%` })));
  const orden = nivel => etiquetas.filter(e => e.nivel === nivel).sort((a, b) => b.probabilidad - a.probabilidad);
  return h("div", { class: "detalle-etiquetas" },
    h("h4", {}, t.categorias), h("ul", {}, orden("categoría").map(fila)),
    h("h4", {}, t.diagnosticos), h("ul", {}, orden("diagnóstico").slice(0, 5).map(fila)),
    h("p", { class: "nota nota-etiquetas" }, t.explicaEtiquetas(iconoAlerta(t.alerta))));
}

// Áreas de interés: un color por zona del modelo, en orden de efecto; los latidos atípicos van punteados
const COLORES_AREAS = ["s1", "s3", "s4"];
const COLOR_LATIDOS = "s2";

function coloresAreas(a) {
  let k = 0;
  return a.grupos.map(gr => (gr.zona ? COLORES_AREAS[k++] : null));
}

function regionesAreas(a, m, colores) {
  const t = tr().areas;
  const pct = v => g.formatear(100 * v, "pct");
  const regiones = [];
  a.grupos.forEach((gr, i) => {
    const filas = [{ valor: `${pct(m.probabilidad)} → ${pct(gr.probabilidad_sin)}`, etiqueta: t.siNormal },
                   ...Object.entries(gr.desvios).map(([d, z]) => ({ valor: d, etiqueta: t.desvio(g.formatear(z, 1)) }))];
    for (const [inicio, fin] of gr.intervalos) {
      regiones.push({ inicio, fin, derivaciones: gr.derivaciones.length === 12 ? null : gr.derivaciones, color: colores[i], titulo: gr.nombre, filas });
    }
  });
  for (const l of a.latidos) {
    regiones.push({ inicio: l.inicio, fin: l.fin, derivaciones: null, color: COLOR_LATIDOS, punteada: true,
                    titulo: tituloLatido(l), filas: l.motivos.map(valor => ({ valor })) });
  }
  return regiones;
}

function tituloLatido(l) {
  const t = tr().areas;
  const s = g.formatear(l.tiempo, 1);
  return l.tipo === "pausa" ? t.pausa(s) : t.latido(s);
}

// Leyenda de lo sombreado: cada zona del modelo con sus derivaciones y cada latido atípico (el detalle va en el tooltip)
function panelAreas(a, colores) {
  const t = tr().areas;
  if (!a.disponible) return [h("p", {}, a.motivo)];
  const muestra = (color, punteada = false) => h("i", {
    class: `muestra-area${punteada ? " punteada" : ""}${color ? "" : " sin-lugar"}`, style: color ? `--area: var(--${color})` : "", "aria-hidden": "true" });
  const donde = gr => (!gr.zona ? "" : gr.derivaciones.length === 12 ? t.todas : t.en(gr.derivaciones.join(", ")));
  const items = [
    ...a.grupos.map((gr, i) => h("li", {}, muestra(colores[i]), h("span", {}, h("b", {}, gr.nombre), " ", donde(gr)))),
    ...a.latidos.map(l => h("li", {}, muestra(COLOR_LATIDOS, true), h("span", {}, tituloLatido(l)))),
  ];
  const aviso = !a.riesgo ? t.sinRiesgo : !a.grupos.length ? t.ninguna : null;
  return [aviso ? h("p", {}, aviso) : null, items.length ? h("ul", { class: "lista-areas" }, items) : null].filter(Boolean);
}

// Frecuencia de muestreo, duración, formato y modelo usado, en un renglón
function lineaFicha(f, m) {
  const t = tr().registro;
  const dato = (etiqueta, valor) => h("span", {}, h("small", {}, etiqueta), " ", valor);
  return h("p", { class: "linea-ficha" },
    dato(t.ficha.fs, `${g.formatear(f.fs)} Hz`), dato(t.ficha.duracion, `${g.formatear(f.duracion)} s`), dato(t.ficha.formato, f.formato),
    m.conectado && m.nombre ? dato(t.modelo, m.nombre) : null);
}

// Una tabla del resultado en su tarjeta; completo: ocupa las dos columnas
function tarjetaTabla(b, completo = false) {
  return h("section", { class: `tarjeta${completo ? " completo" : ""}` },
    h("div", { class: "tarjeta-cabeza" }, h("h3", {}, b.titulo)), h("div", { class: "cuerpo" }, tabla(b.datos)));
}

function mostrarRegistro(destino, r, conTitulo = true) {
  const t = tr().registro;
  const f = r.ficha;
  const papel = h("div", { class: "cuerpo" });
  // El trazo siempre está en mV: los archivos en otra unidad se convierten al leerlos
  const unidad = h("span", { class: "unidad-ecg" }, `${t.ficha.unidades}: mV${f.unidades === "mV" ? "" : ` (${t.archivoEn(f.unidades)})`}`);
  const tituloECG = h("h3", {}, f.duracion > 10 ? `${t.ecg} ${t.primeros}` : t.ecg, unidad);
  // Los análisis guardados antes de las áreas de interés, o sin modelo, no traen "areas": no hay botón
  const colores = r.areas?.disponible ? coloresAreas(r.areas) : [];
  const botonAreas = r.areas ? h("button", { type: "button", class: "boton-secundario boton-areas" }) : null;
  const panel = h("div", { class: "panel-areas" });
  const pintarECG = () => {
    const ver = Boolean(r.areas) && estado.areas;
    if (botonAreas) {
      botonAreas.textContent = ver ? tr().areas.ocultar : tr().areas.mostrar;
      botonAreas.setAttribute("aria-pressed", String(ver));
    }
    panel.hidden = !ver;
    panel.replaceChildren(...(ver ? panelAreas(r.areas, colores) : []));
    dibujarECG(papel, r.ecg, ver && r.areas.disponible ? regionesAreas(r.areas, r.modelo, colores) : []);
  };
  botonAreas?.addEventListener("click", () => {
    estado.areas = !estado.areas;
    pintarECG();
  });
  // Multietiqueta: la tarjeta del modelo, larga, junto a la columna de las dos tablas, ambas del mismo alto.
  // Binario o sin modelo: la tarjeta, corta, junto a las medidas, y la comparación debajo a todo el ancho.
  const [medidas, comparacion] = r.bloques;
  const multi = Boolean(r.modelo.etiquetas);
  const avisos = r.avisos.length
    ? h("section", { class: `tarjeta avisos${multi ? "" : " completo"}` }, h("h3", {}, t.avisos), h("ul", {}, r.avisos.map(a => h("li", {}, a))))
    : null;
  const junto = multi
    ? [h("div", { class: "columna-lateral" }, tarjetaTabla(medidas), tarjetaTabla(comparacion), avisos)]
    : [tarjetaTabla(medidas), tarjetaTabla(comparacion, true), avisos];
  // En el historial el renglón del archivo va en la cabecera, debajo de la fecha (paginaAnalisis)
  destino.replaceChildren(
    conTitulo ? h("h2", { class: "subtitulo" }, `${t.resultado}: ${f.archivo}`) : "",
    conTitulo ? lineaFicha(f, r.modelo) : "",
    h("div", { class: "bloques" },
      tarjetaModelo(r.modelo),
      ...junto,
      h("section", { class: "tarjeta completo" }, h("div", { class: "tarjeta-cabeza" }, tituloECG, botonAreas), papel, panel)));
  estado.dibujo = pintarECG;
  estado.dibujo();
}

// Historial: tabla de los ECGs analizados y cada análisis guardado
const fechaLocal = iso => new Date(iso).toLocaleString(LOCALES[estado.idioma], { dateStyle: "medium", timeStyle: "short" });

function resultadoCorto(f) {
  const t = tr().historial;
  if (f.probabilidad === null) return h("span", { class: "apagado" }, f.modelo_conectado ? t.sinResultado : t.sinModelo);
  const pct = g.formatear(100 * f.probabilidad, "pct");
  return h("span", { class: "resultado-corto" },
    h("span", { class: `estado ${f.riesgo ? "con-riesgo" : "sin-riesgo"}` },
      h("i", { "aria-hidden": "true" }), f.riesgo ? t.riesgo(pct) : t.sinRiesgo(pct)),
    f.modelo_nombre ? h("small", { class: "apagado" }, f.modelo_nombre) : null);
}

function tablaHistorial(filas) {
  const c = tr().historial.columnas;
  const encabezados = [[c.fecha, ""], [c.archivo, "texto"], [c.formato, "texto"], [c.duracion, ""], [c.fc, ""], [c.avisos, ""], [c.resultado, "texto"]];
  return h("table", { class: "tabla-historial" },
    h("thead", {}, h("tr", {}, encabezados.map(([texto, clase]) => h("th", { class: clase }, texto)))),
    h("tbody", {}, filas.map(f => {
      const fila = h("tr", { class: "fila-enlace" },
        h("td", {}, h("a", { href: `#/historial/${f.id}` }, fechaLocal(f.fecha))),
        h("td", { class: "texto" }, f.archivo),
        h("td", { class: "texto" }, f.formato),
        h("td", {}, `${g.formatear(f.duracion)} s`),
        h("td", {}, g.formatear(f.frecuencia_cardiaca, 0)),
        h("td", {}, g.formatear(f.avisos, "entero")),
        h("td", { class: "texto" }, resultadoCorto(f)));
      fila.addEventListener("click", evento => {   // toda la fila abre el análisis, no solo la fecha
        if (evento.target.tagName !== "A") location.hash = `#/historial/${f.id}`;
      });
      return fila;
    })));
}

async function paginaHistorial() {
  const cuerpo = h("div", { class: "bloques" });
  contenido.replaceChildren(...encabezado("historial"), cuerpo);
  const pagina = estado.pagina;
  const filas = await api("/api/historial", new URLSearchParams());
  if (pagina !== estado.pagina) return;
  cuerpo.replaceChildren(filas.length
    ? h("section", { class: "tarjeta completo" }, h("div", { class: "cuerpo" }, tablaHistorial(filas)))
    : h("p", { class: "aviso" }, tr().historial.vacio));
}

async function paginaAnalisis(id) {
  const t = tr().historial;
  const migas = h("p", { class: "migas" }, tr().paginas.historial[0]);
  const titulo = h("h1", {});
  const bajada = h("p", { class: "bajada" });
  const ficha = h("div", {});
  const eliminar = h("button", { type: "button", class: "boton-secundario boton-peligro", hidden: "" }, t.eliminar);
  const resultado = h("div", { class: "registro-resultado" });
  contenido.replaceChildren(migas, h("div", { class: "cabeza-analisis" }, h("div", {}, titulo, bajada, ficha), eliminar), resultado);
  eliminar.addEventListener("click", () => confirmarEliminar(id));
  const pagina = estado.pagina;
  try {
    const r = await api(`/api/historial/${id}`, new URLSearchParams());
    if (pagina !== estado.pagina) return;
    migas.textContent = `${tr().paginas.historial[0]}  ›  ${r.ficha.archivo}`;
    titulo.textContent = r.ficha.archivo;
    bajada.textContent = t.analizado(fechaLocal(r.fecha));
    ficha.replaceChildren(lineaFicha(r.ficha, r.modelo));
    eliminar.hidden = false;
    mostrarRegistro(resultado, r, false);
  } catch (error) {
    if (pagina === estado.pagina) resultado.replaceChildren(h("p", { class: "aviso error" }, error.message));
  }
}

function confirmarEliminar(id) {
  const t = tr().historial;
  const cancelar = h("button", { type: "button", class: "boton-secundario" }, t.cancelar);
  const formulario = h("form", { method: "dialog", class: "dialogo" },
    h("h2", {}, t.eliminar),
    h("p", {}, t.confirmarEliminar),
    h("div", { class: "dialogo-botones" }, cancelar, h("button", { type: "submit", class: "boton boton-rojo" }, t.eliminarSi)));
  const dialogo = document.getElementById("dialogo-registro");
  dialogo.replaceChildren(formulario);
  cancelar.addEventListener("click", () => dialogo.close());
  formulario.addEventListener("submit", async evento => {
    evento.preventDefault();
    dialogo.close();
    const respuesta = await fetch(`/api/historial/${id}`, { method: "DELETE" });
    if (!respuesta.ok && respuesta.status !== 404) return;
    if (estado.ultimoId === id) estado.ultimoId = null;
    location.hash = "#/historial";
  });
  dialogo.showModal();
  cancelar.focus();
}

// Acerca de > Datasets
function filtros(alCambiar) {
  const chips = [];
  const refrescar = () => chips.forEach(c => c.setAttribute("aria-pressed", String(estado.seleccion.includes(c.dataset.nombre))));
  const cambiar = nueva => {
    estado.seleccion = todos().filter(d => nueva.includes(d));
    guardar("seleccion", estado.seleccion);
    refrescar();
    alCambiar();
  };
  const chip = (dataset, clase = "") => {
    const c = h("button", { type: "button", class: `chip ${clase}`, "data-nombre": dataset }, nombre(dataset));
    c.addEventListener("click", () => cambiar(estado.seleccion.includes(dataset)
      ? estado.seleccion.filter(d => d !== dataset)
      : [...estado.seleccion, dataset]));
    chips.push(c);
    return c;
  };
  const t = tr().filtros;
  const accion = (texto, seleccion) => h("button", { type: "button", class: "accion", onclick: () => cambiar(seleccion()) }, texto);
  const fila = h("div", { class: "filtros", role: "group", "aria-label": t.grupo },
    h("span", { class: "filtros-etiqueta" }, t.originales),
    estado.originales.map(d => chip(d)),
    h("span", { class: "separador" }),
    h("span", { class: "filtros-etiqueta" }, t.nuevo),
    chip(estado.nuevo, "chip-nuevo"),
    h("span", { class: "separador" }),
    accion(t.todosOriginales, () => [...estado.originales]),
    accion(t.soloNuevo, () => [estado.nuevo]),
    accion(t.todos, todos));
  refrescar();
  return fila;
}

function paginaEda(ruta) {
  const bloques = h("div", { class: "bloques" });
  contenido.replaceChildren(...encabezado(ruta), filtros(() => cargarEda(ruta, bloques)), bloques);
  cargarEda(ruta, bloques);
}

async function cargarEda(ruta, destino) {
  const id = ++estado.peticion;
  escribirURL(ruta, true);
  if (!estado.seleccion.length) {
    estado.dibujo = null;
    destino.replaceChildren(h("p", { class: "aviso" }, tr().elige));
    return;
  }
  if (!destino.childElementCount) destino.append(h("p", { class: "aviso" }, tr().calculando));
  destino.classList.add("cargando");
  try {
    const bloques = await api(`/api/${ruta}`, new URLSearchParams(estado.seleccion.map(d => ["ds", d])));
    if (id !== estado.peticion) return;
    estado.dibujo = () => dibujarBloques(destino, bloques);
    estado.dibujo();
  } catch (error) {
    if (id !== estado.peticion) return;
    destino.replaceChildren(h("p", { class: "aviso error" }, error.message));
  } finally {
    if (id === estado.peticion) destino.classList.remove("cargando");
  }
}

function paginaVisor() {
  const t = tr().visor;
  const elegirDataset = h("select", {}, todos().map(d => h("option", { value: d }, nombre(d))));
  elegirDataset.value = estado.seleccion[0] ?? todos()[0];
  const elegirClase = h("select", {}, Object.entries(t.clases).map(([valor, texto]) => h("option", { value: valor }, texto)));
  const otro = h("button", { type: "button", class: "boton" }, t.otro);
  const ficha = h("div", { class: "ficha" });
  const papel = h("div", { class: "cuerpo" });
  const tarjetaEcg = h("section", { class: "tarjeta completo" },
    h("div", { class: "tarjeta-cabeza" }, h("div", {}, h("h3", {}, t.titulo), h("p", { class: "nota" }, t.nota))),
    papel);
  const campo = (etiqueta, valor, clase = "") => h("div", { class: clase }, h("small", {}, etiqueta), valor);

  const cargar = async () => {
    const id = ++estado.peticion;
    tarjetaEcg.classList.add("cargando");
    const parametros = new URLSearchParams({ dataset: elegirDataset.value });
    if (elegirClase.value) parametros.set("clase", elegirClase.value);
    try {
      const d = await api("/api/ecg", parametros);
      if (id !== estado.peticion) return;
      ficha.replaceChildren(
        campo(t.dataset, d.dataset === d.origen ? d.dataset : t.de(d.dataset, d.origen)),
        campo(t.registro, d.registro),
        campo(t.edad, d.edad === null ? t.sinDato : t.anios(Math.round(d.edad))),
        campo(t.sexo, t.sexos[d.sexo] ?? t.sinDato),
        campo(t.clase, d.clase ?? t.sinDiagnostico),
        campo(t.disponibles, g.formatear(d.ecgs_disponibles, "entero")),
        campo(t.cardiopatias, d.cardiopatias.length ? d.cardiopatias.join(", ") : t.ninguna, "ancho"),
        campo(t.etiquetas, d.diagnosticos?.length ? d.diagnosticos.join(", ") : t.sinDato, "ancho"));
      estado.dibujo = () => dibujarECG(papel, d);
      estado.dibujo();
    } catch (error) {
      if (id !== estado.peticion) return;
      ficha.replaceChildren(h("p", { class: "aviso error" }, error.message));
      papel.replaceChildren();
    } finally {
      tarjetaEcg.classList.remove("cargando");
    }
  };
  elegirDataset.addEventListener("change", cargar);
  elegirClase.addEventListener("change", cargar);
  otro.addEventListener("click", cargar);

  contenido.replaceChildren(
    ...encabezado("visor"),
    h("div", { class: "filtros controles" }, h("label", {}, t.dataset, elegirDataset), h("label", {}, t.clase, elegirClase), otro),
    h("div", { class: "bloques" }, h("section", { class: "tarjeta completo" }, ficha), tarjetaEcg));
  cargar();
}

// Acerca de > Modelo entrenado
async function paginaModelo() {
  const t = tr().paginaModelo;
  const bloques = h("div", { class: "bloques" });
  const contrato = h("section", { class: "tarjeta completo" },
    h("h3", {}, t.contrato),
    h("dl", { class: "contrato" }, t.partes.map(([termino, definicion]) => [h("dt", {}, termino), h("dd", {}, definicion)])));
  contenido.replaceChildren(...encabezado("modelo"), bloques);
  const pagina = estado.pagina;
  const lista = await api("/api/modelos", new URLSearchParams());
  if (pagina !== estado.pagina) return;
  if (!lista.modelos.length) {
    bloques.replaceChildren(h("p", { class: "aviso" }, t.sinModelo), contrato);
    return;
  }
  modeloValido(lista);
  const botones = h("div", { class: "selector-modelo", role: "group", "aria-label": t.elegir });
  const eleccion = h("section", { class: "tarjeta completo" }, h("h3", {}, t.elegir), botones, h("p", { class: "nota" }, t.elegirNota));
  const cargar = async () => {
    botones.replaceChildren(...lista.modelos.map(m => h("button", {
      type: "button", class: m.id === estado.modelo ? "boton" : "boton-secundario", "aria-pressed": String(m.id === estado.modelo),
      onclick: () => { estado.modelo = m.id; guardar("modelo", m.id); cargar(); },
    }, m.resumen)));
    const pedido = ++estado.peticion;
    const d = await api("/api/modelo", new URLSearchParams({ cual: estado.modelo }));
    if (pagina !== estado.pagina || pedido !== estado.peticion) return;
    estado.dibujo = () => {
      dibujarBloques(bloques, [...lista.comparacion, ...d.bloques]);
      bloques.prepend(eleccion);
      bloques.append(contrato);
    };
    estado.dibujo();
  };
  cargar();
}

// Acerca de > Modelos: modelos probados, resultados y por qué se eligieron los de la UI
async function paginaExperimentos(ruta) {
  const bloques = h("div", { class: "bloques" }, h("p", { class: "aviso" }, tr().calculando));
  contenido.replaceChildren(...encabezado(ruta), bloques);
  const pagina = estado.pagina;
  try {
    const d = await api(`/api/${ruta}`, new URLSearchParams());
    if (pagina !== estado.pagina) return;
    estado.dibujo = () => dibujarBloques(bloques, d);
    estado.dibujo();
  } catch (error) {
    if (pagina === estado.pagina) bloques.replaceChildren(h("p", { class: "aviso error" }, error.message));
  }
}

async function iniciar() {
  const respuesta = await fetch("/api/datasets");
  const d = await respuesta.json();
  estado.originales = d.originales;
  estado.nuevo = d.nuevo;
  estado.modelo = leer("modelo");
  const guardada = leer("seleccion");
  estado.seleccion = Array.isArray(guardada) ? todos().filter(x => guardada.includes(x)) : [];
  if (!estado.seleccion.length) estado.seleccion = [...estado.originales];

  // Idioma: el de la URL, si no el guardado, si no el del navegador, si no español
  const navegador = (navigator.language ?? "").slice(0, 2);
  const enURL = new URLSearchParams(location.hash.split("?")[1] ?? "").get("lang");
  const idioma = [enURL, leer("idioma"), navegador].find(x => IDIOMAS.includes(x)) ?? "es";

  construirIdiomas();
  iniciarTema();
  const archivo = document.getElementById("archivo-ecg");
  archivo.addEventListener("change", () => {
    if (archivo.files.length) pedirFrecuencia([...archivo.files]);
    archivo.value = "";   // permite elegir el mismo archivo otra vez
  });
  aplicarIdioma(idioma);
  document.getElementById("boton-menu").addEventListener("click", () => abrirMenuMovil(true));
  document.getElementById("cortina").addEventListener("click", () => abrirMenuMovil(false));
  let espera;
  addEventListener("resize", () => { clearTimeout(espera); espera = setTimeout(redibujar, 200); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redibujar);
  addEventListener("hashchange", mostrar);
  mostrar();
}

iniciar();
