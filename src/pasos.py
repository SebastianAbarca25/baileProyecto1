from src.peg_solitaire_problem import PegSolitaireProblem, board_str

# Configurar el problema
prob = PegSolitaireProblem(goal='one', algorithm='astar')

# Ejecutar el algoritmo
res = prob.solve()

# Mostrar solo los primeros 5 tableros, aunque no llegue a la meta
if "boards" in res:
    for i, board in enumerate(res["boards"][:5]):  # mostrar 5 primeros pasos
        print(f"\nPaso {i}:")
        print(board_str(board))
else:
    print("No se generó un camino con tableros.")
