# Problema de búsqueda: Peg Solitaire (Come Solo)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Dict, Any, Callable
from heapq import heappush, heappop
import time
import sys
import collections

Board = List[List[int]]           # -1 fuera, 0 vacío, 1 ficha
Move  = Tuple[Tuple[int,int], Tuple[int,int]]  # (i,j)->(i2,j2)
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

def clone(b: Board) -> Board:
    return [row[:] for row in b]

def in_bounds(b: Board, i: int, j: int) -> bool:
    return 0 <= i < 7 and 0 <= j < 7 and b[i][j] != -1

def count_pegs(b: Board) -> int:
    return sum(1 for i in range(7) for j in range(7) if b[i][j]==1)

def board_str(b: Board) -> str:
    s=''
    for i in range(7):
        s += ' '.join({-1:' ', 0:'.', 1:'o'}[b[i][j]] for j in range(7)) + "\n"
    return s

def apply_move(b: Board, m: Move) -> Board:
    (i,j),(i2,j2) = m
    di, dj = (i2 - i) // 2, (j2 - j) // 2
    nb = clone(b)
    nb[i][j] = 0
    nb[i+di][j+dj] = 0
    nb[i2][j2] = 1
    return nb

def state_key(b: Board) -> str:
    # cadena sin '-' (sumamos 1 a cada celda)
    return ''.join(''.join(str(c+1) for c in row) for row in b)

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

class PegSolitaireProblem:
    """
    Clase de problema de búsqueda en el mismo estilo OO:
    - Selección de algoritmo por nombre en __init__ (getattr).
    - Métodos públicos: bfs, dfs, astar, successors, is_goal, heuristic...
    - solve() invoca el algoritmo seleccionado.
    """

    def __init__(self,
                 goal: str = 'one',            # 'one' o 'center'
                 heuristic_name: str = 'pegs', # 'pegs' (admisible) o 'pegs_center_bias' (no admisible)
                 algorithm: str = 'astar',     # 'astar' | 'bfs' | 'dfs'
                 depth_limit: int = 64):
        self.goal = goal
        self.heuristic_name = heuristic_name
        self.algorithm_name = algorithm
        self.depth_limit = depth_limit

        # Estado/informes
        self.solution_: Dict[str, Any] = {}
        self.expanded_ : int = 0
        self.depth_   : Optional[int] = None
        self.time_    : float = 0.0
        self.path_    : Optional[List[str]] = None

        # Algoritmo seleccionado por nombre
        self.algorithm = getattr(self, algorithm)

    # ---------- Definición del problema ----------
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
                        action = f"(({i},{j})->({i2},{j2}))"
                        yield (action, nb, 1)

    def is_goal(self, b: Board) -> bool:
        if self.goal == 'center':
            return count_pegs(b)==1 and b[3][3]==1
        return count_pegs(b)==1

    def heuristic(self, b: Board) -> float:
        # h admisible: pegs-1
        if self.heuristic_name == 'pegs':
            return max(0, count_pegs(b)-1)
        # no admisible: pegs-1 + lambda*dist_min_al_centro (sesgo hacia centro)
        if self.heuristic_name == 'pegs_center_bias':
            lam = 0.25
            D=[]
            for i in range(7):
                for j in range(7):
                    if b[i][j]==1:
                        D.append(abs(i-3)+abs(j-3))
            dist = min(D) if D else 0
            return (count_pegs(b)-1) + lam*dist
        # por defecto, admisible
        return max(0, count_pegs(b)-1)

    # ---------- Algoritmos ----------
    def bfs(self, start: Optional[Board]=None) -> Dict[str,Any]:
        t0 = time.time()
        if start is None: start = self.initial_state()
        nodes = [Node(start,0,None,None)]
        q = collections.deque([0])
        seen = {state_key(start): 0}
        expanded = 0
        while q:
            idx = q.popleft()
            n = nodes[idx]
            expanded += 1
            if self.is_goal(n.board):
                res = {'found': True,
                       'path': reconstruct_path(nodes, idx),
                       'time': time.time()-t0,
                       'expanded': expanded,
                       'depth': n.g}
                self._stash(res)
                return res
            for act, nb, cost in self.successors(n.board):
                k = state_key(nb)
                if k not in seen:
                    seen[k]=1
                    nodes.append(Node(nb, n.g+cost, act, idx))
                    q.append(len(nodes)-1)
        res = {'found': False, 'time': time.time()-t0, 'expanded': expanded}
        self._stash(res)
        return res

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
            if n.g >= self.depth_limit:
                return False
            key = state_key(n.board)
            seen.add(key)
            for act, nb, cost in self.successors(n.board):
                k = state_key(nb)
                if k in seen: 
                    continue
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
        self._stash(res)
        return res

    def astar(self, start: Optional[Board]=None) -> Dict[str,Any]:
        t0 = time.time()
        if start is None: start = self.initial_state()
        nodes = [Node(start,0,None,None)]
        expanded = 0
        openpq: List[Tuple[float,int]] = []
        heappush(openpq, (self.heuristic(start), 0))
        best_g: Dict[str,int] = {state_key(start): 0}
        while openpq:
            _, idx = heappop(openpq)
            n = nodes[idx]
            expanded += 1
            if self.is_goal(n.board):
                res = {'found': True,
                       'path': reconstruct_path(nodes, idx),
                       'time': time.time()-t0,
                       'expanded': expanded,
                       'depth': n.g}
                self._stash(res)
                return res
            for act, nb, cost in self.successors(n.board):
                g2 = n.g + cost
                k = state_key(nb)
                if k not in best_g or g2 < best_g[k]:
                    best_g[k] = g2
                    nodes.append(Node(nb, g2, act, idx))
                    f = g2 + self.heuristic(nb)
                    heappush(openpq, (f, len(nodes)-1))
        res = {'found': False, 'time': time.time()-t0, 'expanded': expanded}
        self._stash(res)
        return res

    # ---------- API ----------
    def solve(self, start: Optional[Board]=None) -> Dict[str,Any]:
        """
        Ejecuta el algoritmo elegido en __init__(algorithm=...).
        Guarda resultados en atributos *_ para consulta/tabla.
        """
        return self.algorithm(start)

    # ---------- Utilidades ----------
    def _stash(self, res: Dict[str,Any]) -> None:
        self.solution_ = res
        self.expanded_ = res.get('expanded', 0)
        self.depth_    = res.get('depth', None)
        self.time_     = res.get('time', 0.0)
        self.path_     = res.get('path', None)

    # Pretty print
    @staticmethod
    def print_board(b: Board) -> None:
        print(board_str(b))

def _safe_pandas_import():
    try:
        import pandas as pd
        return pd
    except Exception:
        return None

def run_experiments(goal: str = 'one',
                    heuristic_name: str = 'pegs',
                    depth_limit: int = 70,
                    show_path_example: bool = False):
    """
    Corre BFS, DFS y A* sobre el mismo problema y devuelve una tabla (pandas.DataFrame si está disponible)
    con: Algoritmo, Encontró?, Tiempo (s), Nodos expandidos, Long. solución (movs).
    """
    #algos = ['bfs', 'dfs', 'astar']
    algos = ['astar']
    rows  = []

    for algo in algos:
        prob = PegSolitaireProblem(goal=goal,
                                   heuristic_name=heuristic_name,
                                   algorithm=algo,
                                   depth_limit=depth_limit)
        s0   = prob.initial_state()
        res  = prob.solve(s0)

        rows.append({
            'Algoritmo': algo.upper(),
            'Encontró?': res.get('found', False),
            'Tiempo (s)': round(res.get('time', float('nan')), 4),
            'Nodos expandidos': res.get('expanded', None),
            'Long. solución (movs)': res.get('depth', None),
        })

    pd = _safe_pandas_import()
    if pd is not None:
        df = pd.DataFrame(rows)
    else:
        # Fallback simple sin pandas
        df = rows

    # (Opcional) imprimir un ejemplo paso a paso de la mejor corrida (A*)
    if show_path_example:
        prob = PegSolitaireProblem(goal=goal,
                                   heuristic_name=heuristic_name,
                                   algorithm='astar',
                                   depth_limit=depth_limit)
        s0 = prob.initial_state()
        res = prob.solve(s0)
        print("\n▶ Ejemplo paso a paso con A*:")
        print("Estado inicial (pegs={}):".format(count_pegs(s0)))
        print(board_str(s0))
        if res.get('found'):
            b = s0
            for t, act in enumerate(res['path'], start=1):
                print(f"Paso {t}: {act}")
                coords = act.replace('(', '').replace(')', '').replace('->', ',').split(',')
                i,j,i2,j2 = map(int, coords)
                b = apply_move(b, ((i,j),(i2,j2)))
                print(board_str(b))
        else:
            print("No se encontró solución con A* en esta configuración.")

    return df


# ───────────────────────────────────────────────────────────────
#  MAIN ampliado: imprime la tabla comparativa
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Configura aquí tus variantes:
    GOAL = 'one'              # 'one' o 'center'
    H    = 'pegs'             # 'pegs' (admisible) o 'pegs_center_bias' (no admisible)
    DLIM = 70                 # límite DFS

    print("== Comparativa BFS / DFS / A* ==")
    tabla = run_experiments(goal=GOAL, heuristic_name=H, depth_limit=DLIM, show_path_example=False)

    print("== Solo A* ==")
    prob = PegSolitaireProblem(goal=GOAL, heuristic_name=H, algorithm='astar', depth_limit=DLIM)
    res = prob.solve()
    print(res)

    try:
        import pandas as pd  # noqa
        print(tabla.to_string(index=False))
    except Exception:
        # Fallback sin pandas
        print("\nAlgoritmo | Encontró? | Tiempo (s) | Nodos expandidos | Long. solución (movs)")
        for r in tabla:
            print("{:<9} | {:<9} | {:<9} | {:<16} | {}".format(
                r['Algoritmo'],
                str(r['Encontró?']),
                r['Tiempo (s)'],
                r['Nodos expandidos'],
                r['Long. solución (movs)']
            ))