import numpy as np
import matplotlib.pyplot as plt

# ==============================
# DATOS (EJEMPLO)
# ==============================
x = np.array([2, 4, 6, 8, 10])
y = np.array([45, 55, 70, 78, 88])
n = len(x)

print("\n")
print("╔════════════════════════════════════════════════════════════════════╗")
print("║ BLOQUE 1: REGRESIÓN LINEAL SIMPLE                                  ║")
print("╚════════════════════════════════════════════════════════════════════╝\n")

print("─────────────────────────────────────────────────────────────────────")
print("Ecuación fundamental: ŷ = a + bx")
print("  • a = intersección (donde la recta cruza el eje y)")
print("  • b = pendiente (cambio en y por cada unidad de x)")
print("  • ŷ = valor predicho (estimado por el modelo)\n")

print("Fórmula de la pendiente: b = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²")
print("Fórmula de la intersección: a = ȳ - b·x̄\n")

print("[D] EJEMPLO APLICADO - MÉTODO 1 (DESDE CERO)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

# ==============================
# PASO 1: MEDIAS
# ==============================
x_media = np.mean(x)
y_media = np.mean(y)

print("PASO 1: Calcular promedios")
print(f"│ x̄ = Σxᵢ / n = {np.sum(x)} / {n} = {x_media:.2f} horas")
print(f"│ ȳ = Σyᵢ / n = {np.sum(y)} / {n} = {y_media:.2f} puntos\n")

# ==============================
# PASO 2: DESVIACIONES
# ==============================
dx = x - x_media
dy = y - y_media
dx_dy = dx * dy
dx_cuad = dx ** 2

print("PASO 2: Construir tabla de desviaciones")
print("│┌─────┬────────────┬────────────┬──────────────┬──────────────┬─────────────┬──────────────┐")
print("││  i  │    xᵢ      │    yᵢ      │  (xᵢ - x̄)   │  (yᵢ - ȳ)   │ (xᵢ-x̄)(yᵢ-ȳ) │   (xᵢ-x̄)²    │")
print("│├─────┼────────────┼────────────┼──────────────┼──────────────┼─────────────┼──────────────┤")

for i in range(n):
    print(f"││ {i+1:<3} │   {x[i]:6.2f}   │   {y[i]:6.2f}   │   "
          f"{dx[i]:8.2f}   │   {dy[i]:8.2f}   │   "
          f"{dx_dy[i]:9.2f}   │   {dx_cuad[i]:8.2f}    │")

print("│├─────┼────────────┼────────────┼──────────────┼──────────────┼─────────────┼──────────────┤")
print(f"││ SUM │   {np.sum(x):6.2f}   │   {np.sum(y):6.2f}   │   "
      f"{np.sum(dx):8.2f}   │   {np.sum(dy):8.2f}   │   "
      f"{np.sum(dx_dy):9.2f}   │   {np.sum(dx_cuad):8.2f}    │")
print("│└─────┴────────────┴────────────┴──────────────┴──────────────┴─────────────┴──────────────┘\n")

# ==============================
# PASO 3: PARÁMETROS
# ==============================
numerador_b = np.sum(dx_dy)
denominador_b = np.sum(dx_cuad)
b = numerador_b / denominador_b
a = y_media - b * x_media

print("PASO 3: Calcular parámetros (a, b)")
print(f"│ b = {numerador_b:.2f} / {denominador_b:.2f} = {b:.4f}")
print(f"│ a = {y_media:.2f} - {b:.4f} × {x_media:.2f} = {a:.4f}\n")

# ==============================
# PASO 4: PREDICCIONES
# ==============================
y_pred = a + b * x

print("PASO 4: Generar predicciones ŷᵢ = a + b·xᵢ")
print("│┌─────┬─────────┬─────────┬─────────────┬─────────────────┐")
print("││  i  │  xᵢ     │   yᵢ    │    ŷᵢ       │  Error (yᵢ-ŷᵢ)  │")
print("│├─────┼─────────┼─────────┼─────────────┼─────────────────┤")

for i in range(n):
    error = y[i] - y_pred[i]
    print(f"││ {i+1:<3} │  {x[i]:5.1f}  │  {y[i]:5.1f}  │   "
          f"{y_pred[i]:7.2f}   │     {error:7.2f}      │")

print("│└─────┴─────────┴─────────┴─────────────┴─────────────────┘\n")

print("RESULTADO - ECUACIÓN FINAL:")
print("┌──────────────────────────────────────────────────────────────┐")
print(f"│  ŷ = {a:.2f} + {b:.2f}·x                                     │")
print("└──────────────────────────────────────────────────────────────┘\n")

# ==============================
# GRÁFICA
# ==============================
plt.figure(figsize=(14, 6))

# Subgráfica 1
plt.subplot(1, 2, 1)
plt.scatter(x, y, s=150)
x_linea = np.linspace(min(x)-1, max(x)+1, 100)
y_linea = a + b * x_linea
plt.plot(x_linea, y_linea, 'r-', linewidth=3)
plt.scatter(x, y_pred, marker='s', s=120)

for i in range(n):
    plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'k--')

plt.xlabel("Horas de Estudio (x)", fontweight='bold')
plt.ylabel("Calificación (y)", fontweight='bold')
plt.title("Regresión Lineal Simple (Método 1)", fontweight='bold')
plt.text(2, 92, f"ŷ = {a:.2f} + {b:.2f}x",
         bbox=dict(facecolor='yellow'))
plt.grid(True)
plt.xlim(0, 12)
plt.ylim(30, 100)

# Subgráfica 2
plt.subplot(1, 2, 2)
plt.axis("off")
texto = (f"COMPONENTES DE LA REGRESIÓN\n\n"
         f"Intersección (a) = {a:.2f}\n"
         f"Pendiente (b) = {b:.2f}\n\n"
         f"Ecuación: ŷ = a + b·x")
plt.text(0.1, 0.5, texto)

plt.suptitle("BLOQUE 1: Regresión Lineal Simple", fontweight='bold')
plt.show()

print("\n✓ Gráfica BLOQUE 1 generada")