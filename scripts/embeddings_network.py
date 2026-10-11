from __future__ import annotations

import argparse
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec

from inf8239_u02.config import ROOT, settings
from inf8239_u02.data import load_dataset

DEFAULT_WINDOW = 5
DEFAULT_MIN_COUNT = 1
VECTOR_SIZE = 60
EPOCHS = 30
SEED = 42


def tokenize(value: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", str(value).lower())


def token_coverage(sentences: list[list[str]], vocabulary) -> float:
    """Proporción de apariciones de tokens (no de palabras distintas) que están en el vocabulario."""
    tokens = [token for sentence in sentences for token in sentence]
    return sum(token in vocabulary for token in tokens) / max(len(tokens), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", default="servicio")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--network-demo", action="store_true")
    args = parser.parse_args()

    df = load_dataset()
    # Mismo criterio que en el entrenamiento del clasificador: textos únicos.
    texts = df[settings.text_column].dropna().drop_duplicates()
    sentences = [tokenize(value) for value in texts]
    model = Word2Vec(
        sentences, vector_size=VECTOR_SIZE, window=args.window, min_count=args.min_count,
        workers=1, seed=SEED, epochs=EPOCHS,
    )
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    is_default = (args.window, args.min_count) == (DEFAULT_WINDOW, DEFAULT_MIN_COUNT)
    suffix = "" if is_default else f"_w{args.window}_mc{args.min_count}"
    model.save(str(ROOT / f"models/word2vec{suffix}.model"))

    coverage = token_coverage(sentences, model.wv)
    in_vocabulary = args.word in model.wv
    neighbors = []
    if in_vocabulary:
        neighbors = [
            (word, round(float(score), 4))
            for word, score in model.wv.most_similar(args.word, topn=min(10, len(model.wv) - 1))
        ]
    report = {
        "word": args.word, "window": args.window, "min_count": args.min_count,
        "vector_size": VECTOR_SIZE, "epochs": EPOCHS, "seed": SEED,
        "documents": len(sentences), "tokens": sum(len(s) for s in sentences),
        "vocabulary": len(model.wv), "coverage": round(coverage, 4),
        "in_vocabulary": in_vocabulary, "neighbors": neighbors,
    }
    report_path = ROOT / f"reports/embeddings_w{args.window}_mc{args.min_count}_{args.word}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Parámetros: window={args.window} min_count={args.min_count}")
    print("Mensajes únicos:", len(sentences), "Tokens:", report["tokens"])
    print("Vocabulario:", report["vocabulary"], "Cobertura:", report["coverage"])
    if in_vocabulary:
        print("Vecinos:", neighbors)
    else:
        print(f"La palabra '{args.word}' no está en el vocabulario")
    print("Reporte guardado en:", report_path)
    if not args.network_demo:
        return

    graph = nx.karate_club_graph()
    metrics = pd.DataFrame({
        "degree": nx.degree_centrality(graph),
        "betweenness": nx.betweenness_centrality(graph),
        "pagerank": nx.pagerank(graph),
    })
    metrics.index.name = "node"
    metrics.to_csv(ROOT / "reports/centralities.csv")
    communities = list(nx.community.greedy_modularity_communities(graph))
    community = {node: index for index, group in enumerate(communities) for node in group}
    print("Nodos:", graph.number_of_nodes(), "Aristas:", graph.number_of_edges())
    print("Comunidades:", len(communities), "Modularidad:", round(nx.community.modularity(graph, communities), 3))
    for name in ("degree", "betweenness", "pagerank"):
        top = metrics[name].sort_values(ascending=False).head(3)
        print(f"Top 3 por {name}:", {int(node): round(float(value), 3) for node, value in top.items()})
    position = nx.spring_layout(graph, seed=42)
    nx.draw(graph, position, node_color=[community[node] for node in graph], cmap="tab10",
            with_labels=True, font_size=7)
    plt.title("Comunidades estructurales · red de demostración")
    plt.savefig(ROOT / "reports/network.png", dpi=180, bbox_inches="tight")
    plt.close()
    nx.write_graphml(graph, ROOT / "reports/social_network.graphml")


if __name__ == "__main__":
    main()
