# src/peg_solitaire_problem.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Dict, Any
from heapq import heappush, heappop
import time, sys

# ──────────────────────────────────────────────
# Configuración de límites
# ──────────────────────────────────────────────
MAX_EXPANSIONS = 10_000_000   # Exploración máxima en A*
TIMEOUT = 900                 # Tiempo máximo (segundos) en A*
PROGRESS_EVERY = 50000        # Cada cuántos nodos imprimir progreso

# ──────────────────────────────────────────────
# Representación del tablero
# ──────────────────────────────────────────────
Board = List[List[int]]           # -1 fuera, 0 vacío, 1 ficha
Move  = Tuple[Tuple[int,int], Tuple[int,int]]
DIRS  = [(1,0),(-1,0),(0,1),(0,-1)]

def english_board() -> Board:
    layout = [
        ['O','O','X','X','X','O','O'],
        ['O','O','X','X','X','O','O'],
        ['X','X','X','X','X','X','X'],
        ['X','X','X','0','X','X','X'],  # centro vacío
        ['X','X','X','X','X','X','X'],
        ['O','O','X','X','X','O','O'],
        ['O','O','X','X','X','O','O'],
    ]
    b: Board = []
    for r in layout:
        row=[]
        for c in r:
            if c=='O': row.append(-1)
            elif c=='0': row.append(0)
            else: row.append(1)
        b.append(row)
    return b

# Casos casi resueltos (para demo rápida)
def almost_solved_center() -> Board:
    b = [[-1]*7 for _ in range(7)]
    b[3][1] = 1
    b[3][2] = 1
    b[3][3] = 0
    return b

def almost_solved_any() -> Board:
    b = [[-1]*7 for _ in range(7)]
    b[3][2] = 1
    b[3][3] = 1
    b[3][4] = 0
    return b

def clone(b: Board) -> Board:
    return [row[:] for row in b]

def in_bounds(b: Board, i: int, j: int) -> bool:
    return 0 <= i < 7 and 0 <= j < 7 and b[i][j] != -1

def count_pegs(b: Board) -> int:
    return sum(1 for i in range(7) for j in range(7) if b[i][j]==1)

def board_str(b: Board) -> str:
    return "\n".join(" ".join({-1:' ', 0:'.', 1:'o'}[c] for c in row) for row in b)

def apply_move(b: Board, m: Move) -> Board:
    (i,j),(i2,j2) = m
    di, dj = (i2 - i) // 2, (j2 - j) // 2
    nb = clone(b)
    nb[i][j] = 0
    nb[i+di][j+dj] = 0
    nb[i2][j2] = 1
    return nb

def state_key(b: Board) -> str:
    return ''.join(''.join(str(c+1) for c in row) for row in b)

# ──────────────────────────────────────────────
# Nodo de búsqueda
# ──────────────────────────────────────────────
@dataclass
class Node:
    board: Board
    g: int
    action: Optional[str]
    parent: Optional[int]

def reconstruct_path(nodes: List[Node], idx: int) -> List[str]:
    path=[]
    while idx is not None:
        n = nodes[idx]
        if n.action is not None: path.append(n.action)
        idx = n.parent
    path.reverse()
    return path

def reconstruct_boards(nodes: List[Node], idx: int) -> List[Board]:
    boards=[]
    while idx is not None:
        n = nodes[idx]
        boards.append(n.board)
        idx = n.parent
    boards.reverse()
    return boards

# ──────────────────────────────────────────────
# Heurísticas
# ──────────────────────────────────────────────
def h_pegs(b: Board) -> float:
    return max(0, count_pegs(b)-1)

def h_center_distance(b: Board) -> float:
    """Número de fichas + distancia Manhattan al centro (3,3)."""
    dist = 0
    for i in range(7):
        for j in range(7):
            if b[i][j] == 1:
                dist += abs(i-3) + abs(j-3)
    return count_pegs(b) - 1 + 0.5*dist

# ──────────────────────────────────────────────
# Clase del problema
# ──────────────────────────────────────────────
class PegSolitaireProblem:
    def __init__(self,
                 goal: str = 'center',
                 heuristic_name: str = 'center_dist',
                 algorithm: str = 'astar',
                 depth_limit: int = 100):
        self.goal = goal
        self.heuristic_name = heuristic_name
        self.algorithm_name = algorithm
        self.depth_limit = depth_limit
        self.solution_: Dict[str, Any] = {}
        self.algorithm = getattr(self, algorithm)

    def initial_state(self) -> Board:
        return english_board()

    def successors(self, b: Board) -> Iterable[Tuple[str, Board, int]]:
        for i in range(7):
            for j in range(7):
                if b[i][j] != 1: continue
                for di,dj in DIRS:
                    i1,j1 = i+di, j+dj
                    i2,j2 = i+2*di, j+2*dj
                    if in_bounds(b,i2,j2) and b[i1][j1]==1 and b[i2][j2]==0:
                        nb = apply_move(b, ((i,j),(i2,j2)))
                        yield (f"(({i},{j})->({i2},{j2}))", nb, 1)

    def is_goal(self, b: Board) -> bool:
        if self.goal == 'center':
            return count_pegs(b)==1 and b[3][3]==1
        return count_pegs(b)==1

    def heuristic(self, b: Board) -> float:
        if self.heuristic_name == 'pegs':
            return h_pegs(b)
        elif self.heuristic_name == 'center_dist':
            return h_center_distance(b)
        return 0

    # DFS
    def dfs(self, start: Optional[Board]=None) -> Dict[str,Any]:
        t0 = time.time()
        if start is None: start = self.initial_state()
        nodes = [Node(start,0,None,None)]
        expanded = 0
        best: Optional[List[str]] = None
        sys.setrecursionlimit(10000)
        seen = set()

        def rec(idx:int) -> bool:
            nonlocal expanded, best
            n = nodes[idx]
            expanded += 1
            if self.is_goal(n.board):
                best = reconstruct_path(nodes, idx)
                return True
            if n.g >= self.depth_limit or expanded >= MAX_EXPANSIONS:
                return False
            key = state_key(n.board)
            seen.add(key)
            for act, nb, cost in self.successors(n.board):
                k = state_key(nb)
                if k in seen: continue
                nodes.append(Node(nb, n.g+cost, act, idx))
                if rec(len(nodes)-1):
                    return True
            seen.discard(key)
            return False

        found = rec(0)
        res = {
            'found': found,
            'path': best if found else None,
            'time': time.time()-t0,
            'expanded': expanded,
            'depth': len(best) if best else None
        }
        self.solution_ = res
        return res

    # A*
    def astar(self, start: Optional[Board]=None) -> Dict[str,Any]:
        t0 = time.time()
        if start is None:
            start = self.initial_state()
        nodes = [Node(start,0,None,None)]
        expanded = 0
        openpq: List[Tuple[float,int]] = []
        heappush(openpq, (self.heuristic(start), 0))
        best_g: Dict[str,int] = {state_key(start): 0}

        while openpq:
            if expanded >= MAX_EXPANSIONS or (time.time()-t0) > TIMEOUT:
                res = {
                    'found': False,
                    'time': round(time.time()-t0,4),
                    'expanded': expanded,
                    'depth': None,
                    'stopped': True
                }
                self.solution_ = res
                return res

            _, idx = heappop(openpq)
            n = nodes[idx]
            expanded += 1

            if expanded % PROGRESS_EVERY == 0:
                print(f"[A*] expanded={expanded}")

            if self.is_goal(n.board):
                path = reconstruct_path(nodes, idx)
                boards = reconstruct_boards(nodes, idx)
                res = {
                    'found': True,
                    'path': path,
                    'boards': boards,
                    'time': round(time.time()-t0,4),
                    'expanded': expanded,
                    'depth': n.g
                }
                self.solution_ = res
                return res

            for act, nb, cost in self.successors(n.board):
                g2 = n.g + cost
                k = state_key(nb)
                if k not in best_g or g2 < best_g[k]:
                    best_g[k] = g2
                    nodes.append(Node(nb, g2, act, idx))
                    f = g2 + self.heuristic(nb)
                    heappush(openpq, (f, len(nodes)-1))

        res = {'found': False, 'time': round(time.time()-t0,4), 'expanded': expanded}
        self.solution_ = res
        return res

    def solve(self, start: Optional[Board]=None) -> Dict[str,Any]:
        return self.algorithm(start)

# ──────────────────────────────────────────────
# Experimentos
# ──────────────────────────────────────────────
def run_experiments(goal: str = 'center',
                    heuristic_name: str = 'center_dist',
                    depth_limit: int = 100):
    algos = ['dfs','astar']
    rows  = []

    for algo in algos:
        prob = PegSolitaireProblem(goal=goal,
                                   heuristic_name=heuristic_name,
                                   algorithm=algo,
                                   depth_limit=depth_limit)
        s0   = prob.initial_state()

        t0 = time.time()
        try:
            res = prob.solve(s0)
        except KeyboardInterrupt:
            res = {'found': False}

        if (time.time()-t0) > TIMEOUT:
            res = {'found': False,
                   'time': round(time.time()-t0,4),
                   'expanded': None,
                   'depth': None,
                   'stopped': True}

        rows.append({
            'Algoritmo': algo.upper(),
            'Encontró?': res.get('found', False),
            'Tiempo (s)': round(res.get('time', time.time()-t0), 4),
            'Nodos expandidos': res.get('expanded', None),
            'Long. solución (movs)': res.get('depth', None),
        })

    return rows

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    GOAL = 'center'        # meta clásica
    H    = 'center_dist'   # heurística mejorada
    DLIM = 100             # suficiente para tablero completo

    print("== Comparativa DFS / A* ==")
    tabla = run_experiments(goal=GOAL, heuristic_name=H, depth_limit=DLIM)

    try:
        import pandas as pd
        df = pd.DataFrame(tabla)
        print(df.to_string(index=False))
    except Exception:
        for r in tabla:
            print(r)

    # Ejemplo rápido garantizado
   
    print("\n== Solución completa desde tablero inicial (si se encuentra) ==")
    prob_full = PegSolitaireProblem(goal='center', heuristic_name=H, algorithm='astar')
    res_full = prob_full.solve(prob_full.initial_state())
    if res_full.get('found'):
        print("Movimientos totales:", len(res_full['path']))
        for step, board in enumerate(res_full['boards']):
            print(f"\nPaso {step}:\n{board_str(board)}")
    else:
        print("No se encontró solución completa en los límites dados.")
