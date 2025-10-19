# src/peg_solitaire_problem.py

# Permite usar anotaciones de tipos que se refieren a la misma clase (futuras referencias)
from __future__ import annotations
# Facilita la creación de clases simples (con atributos automáticos)
from dataclasses import dataclass
# Soporte de anotaciones de tipo: listas, tuplas, diccionarios, etc.
from typing import List, Tuple, Iterable, Optional, Dict, Any
# heapq nos da la cola de prioridad necesaria para implementar A*
from heapq import heappush, heappop
# Librerías estándar
import time, sys

# CONFIGURACIÓN DE LÍMITES
MAX_EXPANSIONS = 10_000_000   # Límite de expansiones en A*
TIMEOUT = 900                 # Tiempo máximo de ejecución (segundos)
PROGRESS_EVERY = 50000        # Cada cuántos nodos expandidos imprimir progreso

# REPRESENTACIÓN DEL TABLERO

# Tablero: 7x7 con casillas válidas y no válidas
Board = List[List[int]]           # Cada celda: -1 = fuera, 0 = vacío, 1 = ficha
Move  = Tuple[Tuple[int,int], Tuple[int,int]]   # Movimiento = origen y destino
DIRS  = [(1,0),(-1,0),(0,1),(0,-1)]             # Posibles direcciones: arriba, abajo, izquierda, derecha

# Tablero clásico en cruz de 33 fichas
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
    # O = fuera (-1), 0 = vacío, X = ficha RECORRE LA MATRIZ EN BUCLE PARA RECONOCER CASILLAS
    for r in layout:
        row=[]
        for c in r:
            if c=='O': row.append(-1)
            elif c=='0': row.append(0)
            else: row.append(1)
        b.append(row)
    return b


# Devuelve una copia del tablero (para no modificar el original)
def clone(b: Board) -> Board:
    return [row[:] for row in b]

# Recorre la matriz 7x7 y chequea si las coordenadas están dentro del tablero y no en una casilla inválida
def in_bounds(b: Board, i: int, j: int) -> bool:
    return 0 <= i < 7 and 0 <= j < 7 and b[i][j] != -1

# Recorre la matriz y cuenta cuántas fichas quedan que es X = 1
def count_pegs(b: Board) -> int:
    return sum(1 for i in range(7) for j in range(7) if b[i][j]==1)

# Convierte el tablero a string para imprimirlo
def board_str(b: Board) -> str:
    return "\n".join(" ".join({-1:' ', 0:'.', 1:'o'}[c] for c in row) for row in b)

# Aplica un movimiento (quita ficha origen, elimina la saltada y pone ficha en destino)
def apply_move(b: Board, m: Move) -> Board:
    (i,j),(i2,j2) = m
    di, dj = (i2 - i) // 2, (j2 - j) // 2   # dirección del salto
    nb = clone(b)
    nb[i][j] = 0         # casilla de origen queda vacía
    nb[i+di][j+dj] = 0   # ficha intermedia eliminada
    nb[i2][j2] = 1       # casilla de destino ahora tiene ficha
    return nb

# Representación única de un estado (para detectar repetidos) 
def state_key(b: Board) -> str:
    return ''.join(''.join(str(c+1) for c in row) for row in b)

# NODO DE BÚSQUEDA
@dataclass
class Node:
    board: Board            # Tablero en este estado
    g: int                  # Coste acumulado (profundidad)
    action: Optional[str]   # Movimiento que llevó a este estado
    parent: Optional[int]   # Índice del nodo padre (para reconstruir solución)

# Reconstruye la secuencia de movimientos desde el nodo inicial hasta la solución
def reconstruct_path(nodes: List[Node], idx: int) -> List[str]:
    path=[]
    while idx is not None:
        n = nodes[idx]
        if n.action is not None: path.append(n.action)
        idx = n.parent
    path.reverse()
    return path

# Reconstruye la secuencia de tableros desde inicio hasta solución
def reconstruct_boards(nodes: List[Node], idx: int) -> List[Board]:
    boards=[]
    while idx is not None:
        n = nodes[idx]
        boards.append(n.board)
        idx = n.parent
    boards.reverse()
    return boards

# HEURÍSTICAS
# Heurística básica: fichas restantes menos 1
#Formula de nuestra heurística h(n) = max(0, número de fichas - 1)
def h_pegs(b: Board) -> float:
    return max(0, count_pegs(b)-1)

# Heurística mejorada Calcular la distancia Manhattan de cada ficha al centro para solucion mas optima
def h_center_distance(b: Board) -> float:
    dist = 0
    for i in range(7):
        for j in range(7):
            if b[i][j] == 1:
                dist += abs(i-3) + abs(j-3)
    return count_pegs(b) - 1 + 0.5*dist

# CLASE DEL PROBLEMA (con BFS, DFS y A*)
class PegSolitaireProblem:
    def __init__(self,
                 goal: str = 'center',              # meta por defecto = solo una ficha en el centro
                 heuristic_name: str = 'center_dist', # heurística por defecto
                 algorithm: str = 'astar',          # algoritmo por defecto
                 depth_limit: int = 100):           # límite de profundidad para DFS
        self.goal = goal
        self.heuristic_name = heuristic_name
        self.algorithm_name = algorithm
        self.depth_limit = depth_limit
        self.solution_: Dict[str, Any] = {}
        self.algorithm = getattr(self, algorithm)   # vincula el nombre del algoritmo a la función

    # Estado inicial = tablero clásico
    def initial_state(self) -> Board:
        return english_board()

    # Genera sucesores (movimientos válidos)
    def successors(self, b: Board) -> Iterable[Tuple[str, Board, int]]:
        for i in range(7):
            for j in range(7):
                if b[i][j] != 1: continue   # solo desde casillas con ficha
                for di,dj in DIRS:
                    i1,j1 = i+di, j+dj       # celda intermedia
                    i2,j2 = i+2*di, j+2*dj   # celda destino
                    if in_bounds(b,i2,j2) and b[i1][j1]==1 and b[i2][j2]==0:
                        nb = apply_move(b, ((i,j),(i2,j2)))
                        yield (f"(({i},{j})->({i2},{j2}))", nb, 1)

    # Verifica condición de meta ose haya una sola ficha en el centro 
    def is_goal(self, b: Board) -> bool:
        if self.goal == 'center':
            return count_pegs(b)==1 and b[3][3]==1
        return count_pegs(b)==1

    # Llama a la heurística seleccionada
    def heuristic(self, b: Board) -> float:
        if self.heuristic_name == 'pegs':
            return h_pegs(b)
        elif self.heuristic_name == 'center_dist':
            return h_center_distance(b)
        return 0

    # ───── DFS ─────
    def dfs(self, start: Optional[Board]=None) -> Dict[str,Any]:
        t0 = time.time()
        if start is None: start = self.initial_state()
        nodes = [Node(start,0,None,None)]
        expanded = 0
        best: Optional[List[str]] = None
        sys.setrecursionlimit(10000)   # aumenta límite de recursión
        seen = set()

        # Función recursiva de DFS
        def rec(idx:int) -> bool:
            nonlocal expanded, best
            n = nodes[idx]
            expanded += 1
            if self.is_goal(n.board):    # condición de meta
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

    # ───── A* ─────
    def astar(self, start: Optional[Board]=None) -> Dict[str,Any]:
        t0 = time.time()
        if start is None:
            start = self.initial_state()
        nodes = [Node(start,0,None,None)]
        expanded = 0
        openpq: List[Tuple[float,int]] = []
        heappush(openpq, (self.heuristic(start), 0))   # cola con (f, índice)
        best_g: Dict[str,int] = {state_key(start): 0}  # mejores costes g

        while openpq:
            # Cortes de seguridad
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

            # Meta alcanzada
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

            # Generar sucesores
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

    # Método general solve() que llama al algoritmo elegido
    def solve(self, start: Optional[Board]=None) -> Dict[str,Any]:
        return self.algorithm(start)


# EXPERIMENTOS (corre DFS y A* y arma tabla comparativa)

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
# MAIN (punto de entrada cuando se corre en terminal)

if __name__ == "__main__":
    GOAL = 'center'        # meta clásica
    H    = 'center_dist'   # heurística mejorada #pegs heristica básica
    DLIM = 100             # límite DFS

    print("== Comparativa DFS / A* ==")
    tabla = run_experiments(goal=GOAL, heuristic_name=H, depth_limit=DLIM)

    try:
        import pandas as pd
        df = pd.DataFrame(tabla)
        print(df.to_string(index=False))
    except Exception:
        for r in tabla:
            print(r)

    # Imprime paso a paso la solución completa
    print("\n== Solución completa desde tablero inicial (si se encuentra) ==")
    prob_full = PegSolitaireProblem(goal='center', heuristic_name=H, algorithm='astar')
    res_full = prob_full.solve(prob_full.initial_state())
    if res_full.get('found'):
        print("Movimientos totales:", len(res_full['path']))
        for step, board in enumerate(res_full['boards']):
            print(f"\nPaso {step}:\n{board_str(board)}")
    else:
        print("No se encontró solución completa en los límites dados.")
