import numpy as np

# Configuración inicial
ITERATION_LIMIT = 30
TOLERANCE = 0.0005

# Inicializar la matriz A y el vector b
A = np.array([[52, 30, 18],
              [20, 50, 30],
              [25, 20, 55]])

b = np.array([4800, 5810, 5690])

# Verificar si la matriz tiene diagonal predominante
def verificar_diagonal_predominante(A):
    for i in range(len(A)):
        suma_fila = sum(abs(A[i, j]) for j in range(len(A)) if i != j)
        if abs(A[i, i]) <= suma_fila:
            print(f"Fila {i + 1}: No cumple con diagonal estrictamente dominante.")
        else:
            print(f"Fila {i + 1}: Cumple con diagonal dominante.")
    print()

# Despejar los elementos de la diagonal: Mx + c
def despejar_elementos_diagonal(A, b):
    M = np.zeros_like(A, dtype=float)
    c = np.zeros_like(b, dtype=float)

    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                M[i, j] = -A[i, j] / A[i, i]
        c[i] = b[i] / A[i, i]

    print("Matriz M:")
    print(M)
    print("Vector c:")
    print(c)
    print()

    return M, c

# Encontrar alfa (indicador de convergencia)
def calcular_alfa(M):
    alfa = np.max(np.sum(np.abs(M), axis=1))
    print(f"Alfa (indicador de convergencia): {alfa}")
    if alfa <= 1:
        print("El sistema converge.")
    else:
        print("El sistema podría converger lentamente o no converger.")
    print()
    return alfa

# Método de iteración con Jacobi
def iterar_jacobi(A, b, tol, iter_limit):
    x = np.zeros_like(b, dtype=float)

    for it_count in range(1, iter_limit + 1):
        x_new = np.zeros_like(x, dtype=float)

        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        # Calcular el error máximo
        error_max = np.max(np.abs(x_new - x))
        print(f"Iteración {it_count}: x = {x_new}, Error máximo = {error_max}")

        if error_max < tol:
            print(f"Convergencia alcanzada en {it_count} iteraciones.")
            break

        x = x_new

    return x

# Verificar si la matriz tiene diagonal dominante
print("1. Verificación de diagonal dominante:")
verificar_diagonal_predominante(A)

# Despejar los elementos de la diagonal
print("2. Despejar los elementos de la diagonal:")
M, c = despejar_elementos_diagonal(A, b)

# Encontrar alfa como indicador de convergencia
print("3. Cálculo del indicador de convergencia alfa:")
alfa = calcular_alfa(M)

# Iterar utilizando el método de Jacobi
print("4. Iterar utilizando Jacobi:")
solucion = iterar_jacobi(A, b, TOLERANCE, ITERATION_LIMIT)

# Mostrar la solución final
print("Solución final:")
print(solucion)

# Calcular el error final
error_final = np.dot(A, solucion) - b
print("Error final:")
print(error_final)