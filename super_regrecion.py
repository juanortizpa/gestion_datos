# ==========================================================
# BLOQUE 1: REGRESIÓN LINEAL SIMPLE (VERSIÓN PROFESIONAL)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

# ----------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ----------------------------------------------------------
def regresion_lineal_simple(x, y, mostrar_tabla=True):
    """
    Calcula regresión lineal simple desde cero.
    
    Parámetros:
        x (array-like): Variable independiente
        y (array-like): Variable dependiente
        mostrar_tabla (bool): Mostrar tabla detallada
        
    Retorna:
        dict con resultados
    """
    
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud")
    
    n = len(x)
    
    # =============================
    # Cálculo de medias
    # =============================
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # =============================
    # Desviaciones
    # =============================
    dx = x - x_mean
    dy = y - y_mean
    
    # =============================
    # Parámetros del modelo
    # =============================
    b = np.sum(dx * dy) / np.sum(dx ** 2)
    a = y_mean - b * x_mean
    
    # =============================
    # Predicciones
    # =============================
    y_pred = a + b * x
    residuos = y - y_pred
    
    # =============================
    # Métricas
    # =============================
    ss_total = np.sum((y - y_mean) ** 2)
    ss_res = np.sum(residuos ** 2)
    r2 = 1 - (ss_res / ss_total)
    
    # =============================
    # Tabla detallada
    # =============================
    tabla = pd.DataFrame({
        "x": x,
        "y": y,
        "x - x̄": dx,
        "y - ȳ": dy,
        "(x - x̄)(y - ȳ)": dx * dy,
        "(x - x̄)²": dx ** 2,
        "ŷ": y_pred,
        "Error": residuos
    })
    
    if mostrar_tabla:
        print("\nTABLA DETALLADA:")
        print(tabla.round(3))
    
    # =============================
    # Resultados
    # =============================
    resultados = {
        "a": a,
        "b": b,
        "r2": r2,
        "tabla": tabla,
        "y_pred": y_pred
    }
    
    return resultados


# ----------------------------------------------------------
# DATOS DE EJEMPLO
# ----------------------------------------------------------
x = [2, 4, 6, 8, 10]
y = [50, 60, 70, 85, 95]

# ----------------------------------------------------------
# EJECUCIÓN
# ----------------------------------------------------------
resultado = regresion_lineal_simple(x, y)

a = resultado["a"]
b = resultado["b"]
r2 = resultado["r2"]
y_pred = resultado["y_pred"]

print("\n==========================================")
print("RESULTADOS DEL MODELO")
print("==========================================")
print(f"Intersección (a): {a:.4f}")
print(f"Pendiente (b):    {b:.4f}")
print(f"Ecuación:         ŷ = {a:.4f} + {b:.4f}x")
print(f"R²:               {r2:.4f}")
print("==========================================\n")


# ----------------------------------------------------------
# VISUALIZACIÓN PROFESIONAL
# ----------------------------------------------------------
plt.figure(figsize=(14, 7))

# Scatter
sns.scatterplot(x=x, y=y, s=150, color="#1f77b4", label="Datos reales")

# Línea de regresión
x_line = np.linspace(min(x)-1, max(x)+1, 100)
y_line = a + b * x_line
plt.plot(x_line, y_line, color="red", linewidth=3, label="Recta de regresión")

# Predicciones
sns.scatterplot(x=x, y=y_pred, marker="s", s=120, 
                color="#d62728", label="Predicciones")

# Líneas de error
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y[i], y_pred[i]], 
             linestyle="--", color="gray", alpha=0.7)

plt.title("Regresión Lineal Simple\nModelo Ajustado desde Cero", 
          fontsize=18, fontweight="bold")

plt.xlabel("Horas de Estudio", fontweight="bold")
plt.ylabel("Calificación", fontweight="bold")

plt.legend()
plt.xlim(0, 12)
plt.ylim(30, 100)

# Cuadro informativo
plt.text(
    0.02, 0.95,
    f"ŷ = {a:.2f} + {b:.2f}x\nR² = {r2:.3f}",
    transform=plt.gca().transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round')
)

plt.tight_layout()
plt.show()

print("✓ Gráfica generada correctamente")