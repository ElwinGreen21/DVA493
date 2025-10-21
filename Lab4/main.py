import math
import os

os.chdir(os.path.dirname(__file__))

def read_graph(filename):
    edges = []
    try:
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                # Hoppa över rubriker och tomma rader
                if len(parts) != 3:
                    continue
                u, v, w = parts
                if not (u.isalpha() and v.isalpha() and w.isdigit()):
                    continue
                w = int(w)
                edges.append((u, v, w))
                edges.append((v, u, w))  # eftersom vägarna är dubbla
    except FileNotFoundError:
        print(f"Filen '{filename}' hittades inte!")
    return edges


def bellman_ford(nodes, edges, source):
    distance = {node: float('inf') for node in nodes}
    predecessor = {node: None for node in nodes}
    distance[source] = 0

    # Upprepa |V|-1 gånger
    for _ in range(len(nodes) - 1):
        for u, v, w in edges:
            if distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
                predecessor[v] = u

    # Kolla efter negativa cykler (borde inte finnas i denna uppgift)
    for u, v, w in edges:
        if distance[u] + w < distance[v]:
            raise ValueError("Grafen innehåller en negativ cykel!")

    return distance, predecessor

def get_path(predecessor, start, goal):
    path = []
    current = start
    while current is not None:
        path.append(current)
        if current == goal:
            break
        current = predecessor[current]
    return path if path[-1] == goal else None

if __name__ == "__main__":
    edges = read_graph("city 1.txt")

    # extrahera alla städer
    nodes = sorted(set([u for u, _, _ in edges] + [v for _, v, _ in edges]))

    # kör Bellman–Ford från destinationen F
    distance, predecessor = bellman_ford(nodes, edges, "F")

    print("Kortaste avstånd till F:")
    for node in nodes:
        print(f"{node}: {distance[node]}")

    print("\nKortaste vägar till F:")
    for node in nodes:
        if node != "F":
            path = get_path(predecessor, node, "F")
            print(f"{node} -> F: {path}")

