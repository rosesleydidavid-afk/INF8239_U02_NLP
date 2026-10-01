# Dataset Card — SMS Spam Collection

## Fuente
- **Nombre:** SMS Spam Collection
- **Responsable/Origen:** UCI Machine Learning Repository (recolectado originalmente por Tiago A. Almeida y José María Gómez Hidalgo)
- **URL:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- **Fecha de descarga:** 2026-09-30
- **Licencia:** Dominio público / CC0
- **Hash SHA-256 (archivo original):** `1e673b1d22d211ad065d63305faf0bb2bf5acd169473d14bd403c9725c1d4551`

## Uso previsto
Clasificación binaria de mensajes SMS en español/inglés como spam o no-spam (ham), con fines exclusivamente académicos para practicar un pipeline de clasificación de texto (TF-IDF + clasificadores lineales).

## Diccionario de datos

| Columna | Tipo | Significado | Categorías válidas |
|---|---|---|---|
| `text` | string | Contenido del mensaje SMS | Texto libre |
| `label` | string | Etiqueta de clasificación | `ham` (no-spam) / `spam` |

## Estadísticas (dataset original, antes de limpieza)
- **Filas:** 5,572
- **Nulos:** 0 en ambas columnas
- **Duplicados de texto:** 403 (7.2% del total)
- **Distribución de clases:** 86.6% ham / 13.4% spam
- **Longitud de texto (caracteres):** mediana 62, p90 156, p99 277, máximo 910

## Transformación realizada
Se eliminaron 403 mensajes duplicados exactos (`drop_duplicates` sobre la columna `text`), guardados en `data/processed/dataset_clean.csv`. El archivo original permanece intacto en `data/raw/dataset.csv`. Justificación: evitar que el mismo mensaje aparezca simultáneamente en entrenamiento y prueba, lo que inflaría artificialmente las métricas (fuga de información por duplicados, según U02.A4).

## Ejemplos anonimizados (5 casos representativos)
*Nota: estos mensajes son de un corpus público de 2011-2012; no contienen nombres propios completos ni datos identificables más allá de lo ya público en el dataset original.*

1. `ham` — "Ok lar... Joking wif u oni..."
2. `spam` — "Free entry in 2 a wkly comp to win FA Cup final tkts..."
3. `ham` — "U dun say so early hor... U c already then say..."
4. `spam` — "WINNER!! As a valued network customer you have been selected..."
5. `ham` — "I'm gonna be home soon and i don't want to talk about this stuff..."

## Sesgos y limitaciones previsibles
- **Idioma:** mensajes mayoritariamente en inglés británico/singapurense (el corpus original proviene de Singapur y el Reino Unido) — un modelo entrenado aquí **no generaliza** a SMS en español dominicano u otro dialecto.
- **Periodo:** mensajes recolectados alrededor de 2011-2012 — las técnicas de spam han evolucionado; palabras clave de esa época pueden no representar spam actual.
- **Plataforma:** solo SMS, no aplica directamente a spam de WhatsApp, email o redes sociales.
- **Desbalance:** 86.6%/13.4% — requiere F1 macro y no solo accuracy para evaluar correctamente.

## Usos prohibidos
No debe usarse para tomar decisiones sobre personas reales (bloqueo de números, sanciones), ni representarse como un filtro de spam listo para producción sin reentrenamiento con datos actuales y locales.