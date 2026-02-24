#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLASE: Regresión Lineal Simple y Métricas de Error (MSE, RMSE, R²)
Adaptado para análisis de precios de combustible en Colombia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# Obtener ruta del directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# ═════════════════════════════════════════════════════════════════════════
#  DATOS GLOBALES DEL CASO DE ESTUDIO
# ═════════════════════════════════════════════════════════════════════════

print('\n')
print('┌────────────────────────────────────────────────────────────────────┐')
print('│ RECARGA DE DATOS DEL CSV                                           │')
print('└────────────────────────────────────────────────────────────────────┘')

# Cargar CSV
csv_path = os.path.join(script_dir, 'precio_mes_combustible_20260223.csv')
df = pd.read_csv(csv_path)

print(f'\n✓ CSV cargado: {df.shape[0]} registros')
print(f'  Columnas: {list(df.columns)}')

# Limpiar y procesar datos
df.columns = df.columns.str.strip().str.replace('"', '')
df['Periodo'] = df['Periodo'].astype(str).str.replace('"', '')
df['Mes'] = df['Mes'].astype(int)
df['Precio'] = pd.to_numeric(df['Precio'], errors='coerce')

# Eliminar valores nulos
df = df.dropna(subset=['Precio'])

# Productos únicos
productos = df['Producto'].unique()
print(f'\n  Productos disponibles: {len(productos)}')
for prod in productos[:5]:
    print(f'    - {prod}')

# ═════════════════════════════════════════════════════════════════════════
#  AGREGAR DATOS POR ESTACIÓN DE SERVICIO
# ═════════════════════════════════════════════════════════════════════════

print('\n┌────────────────────────────────────────────────────────────────────┐')
print('│ AGREGACIÓN DE DATOS POR ESTACIÓN DE SERVICIO                       │')
print('└────────────────────────────────────────────────────────────────────┘\n')

# Seleccionar producto principal
producto_analisis = 'GASOLINA CORRIENTE OXIGENADA'

# Filtrar por producto
df_producto = df[df['Producto'] == producto_analisis].copy()

# Agregar por estación de servicio (nombre comercial + ubicación)
datos_agregados = df_producto.groupby('Nombre_comercial').agg({
    'Precio': ['mean', 'count', 'std'],
    'Municipio': 'first'
}).reset_index()

datos_agregados.columns = ['Estacion', 'Precio_Promedio', 'Cantidad_Registros', 'Desv_Std', 'Municipio']
datos_agregados = datos_agregados[datos_agregados['Cantidad_Registros'] >= 2].reset_index(drop=True)
datos_agregados = datos_agregados.sort_values('Cantidad_Registros', ascending=False)

print(f'Análisis para: {producto_analisis}\n')
print(f'Estaciones de servicio: {len(datos_agregados)}')
print(f'Muestra de datos:')
print(datos_agregados.head(10).to_string(index=False))

# Variable independiente (x): número de registros por estación (volumen de transacciones)
# Variable dependiente (y): precio promedio
x = datos_agregados['Cantidad_Registros'].values.astype(float)
y = datos_agregados['Precio_Promedio'].values.astype(float)
n = len(x)

print(f'\nVector x (Registros):      {x}')
print(f'Vector y (Precio prom.): {np.round(y, 2)}')
print(f'Número de observaciones: n = {n}\n')


# ╔════════════════════════════════════════════════════════════════════════╗
#  ║            BLOQUE 1: REGRESIÓN LINEAL SIMPLE                         ║
#  ╚════════════════════════════════════════════════════════════════════════╝

print('╔════════════════════════════════════════════════════════════════════╗')
print('║ BLOQUE 1: REGRESIÓN LINEAL SIMPLE                                  ║')
print('╚════════════════════════════════════════════════════════════════════╝\n')

print('─────────────────────────────────────────────────────────────────────')
print('Ecuación fundamental: ŷ = a + bx')
print('  • a = intersección (donde la recta cruza el eje y)')
print('  • b = pendiente (cambio en y por cada unidad de x)')
print('  • ŷ = valor predicho (estimado por el modelo)\n')

print('Fórmula de la pendiente: b = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²')
print('Fórmula de la intersección: a = ȳ - b·x̄\n')

print('[D] EJEMPLO APLICADO - MÉTODO 1 (DESDE CERO)')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n')

# Cálculo de medias
x_media = np.sum(x) / n
y_media = np.sum(y) / n

print('PASO 1: Calcular promedios')
print(f'│ x̄ = Σxᵢ / n = {np.sum(x):.1f} / {n} = {x_media:.4f} registros')
print(f'│ ȳ = Σyᵢ / n = {np.sum(y):.2f} / {n} = {y_media:.2f} precio\n')

# Cálculo de desviaciones
dx = x - x_media
dy = y - y_media
dx_dy = dx * dy
dx_cuad = dx ** 2

print('PASO 2: Construir tabla de desviaciones')
print('│┌─────┬────────┬─────────────┬──────────────┬──────────────┬─────────────┬──────────────┐')
print('││  i  │  xᵢ    │    yᵢ       │  (xᵢ - x̄)   │  (yᵢ - ȳ)   │ (xᵢ-x̄)(yᵢ-ȳ) │   (xᵢ-x̄)²    │')
print('│├─────┼────────┼─────────────┼──────────────┼──────────────┼─────────────┼──────────────┤')

for i in range(n):
    print(f'││ {i+1:2d}  │ {x[i]:6.2f} │  {y[i]:8.2f}   │   {dx[i]:8.2f}   │   {dy[i]:8.2f}   │   {dx_dy[i]:9.2f}   │   {dx_cuad[i]:8.2f}    │')

print('│├─────┼────────┼─────────────┼──────────────┼──────────────┼─────────────┼──────────────┤')
print(f'││ SUM │{np.sum(x):7.2f} │  {np.sum(y):8.2f}   │   {np.sum(dx):8.2f}   │   {np.sum(dy):8.2f}   │   {np.sum(dx_dy):9.2f}   │   {np.sum(dx_cuad):8.2f}    │')
print('│└─────┴────────┴─────────────┴──────────────┴──────────────┴─────────────┴──────────────┘\n')

# Cálculo de parámetros
numerador_b = np.sum(dx_dy)
denominador_b = np.sum(dx_cuad)
b = numerador_b / denominador_b
a = y_media - b * x_media

print('PASO 3: Calcular parámetros (a, b)')
print(f'│ b = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)² = {numerador_b:.2f} / {denominador_b:.2f} = {b:.4f}')
print(f'│ a = ȳ - b·x̄ = {y_media:.2f} - {b:.4f} × {x_media:.2f} = {a:.4f}\n')

# Predicciones
y_pred = a + b * x

print('PASO 4: Generar predicciones ŷᵢ = a + b·xᵢ')
print('│┌─────┬─────────┬─────────┬─────────────┬─────────────────┐')
print('││  i  │  xᵢ     │   yᵢ    │    ŷᵢ       │  Error (yᵢ-ŷᵢ)  │')
print('│├─────┼─────────┼─────────┼─────────────┼─────────────────┤')

for i in range(n):
    error = y[i] - y_pred[i]
    print(f'││ {i+1:2d}  │ {x[i]:7.2f} │ {y[i]:7.2f} │   {y_pred[i]:7.2f}   │     {error:7.2f}      │')

print('│└─────┴─────────┴─────────┴─────────────┴─────────────────┘\n')

print('RESULTADO - ECUACIÓN FINAL:')
print('┌────────────────────────────────────────────────────────────────────┐')
print(f'│  ŷ = {a:.2f} + {b:.4f}·x                                      │')
print('└────────────────────────────────────────────────────────────────────┘\n')


# ╔════════════════════════════════════════════════════════════════════════╗
#  ║            BLOQUE 2: MSE Y RMSE                                       ║
#  ╚════════════════════════════════════════════════════════════════════════╝

print('\n╔════════════════════════════════════════════════════════════════════╗')
print('║ BLOQUE 2: CUANTIFICACION DEL ERROR - MSE Y RMSE                    ║')
print('╚════════════════════════════════════════════════════════════════════╝\n')

print('FORMALIZACION TECNICA')
print('Residuo: eᵢ = yᵢ - ŷᵢ')
print('MSE = Σ(eᵢ)² / n')
print('RMSE = sqrt(MSE)\n')

print('[D] EJEMPLO APLICADO')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n')

# Cálculo de residuos
residuos = y - y_pred
residuos_cuad = residuos ** 2

print('PASO 1: Calcular residuos (errores de prediccion)')
print('│┌─────┬──────────┬──────────┬──────────────┬────────────────┐')
print('││  i  │    yᵢ    │   y_pred │   eᵢ = yᵢ-ŷᵢ │    eᵢ² (multa)  │')
print('│├─────┼──────────┼──────────┼──────────────┼────────────────┤')

for i in range(n):
    print(f'││ {i+1:2d}  │  {y[i]:8.2f}  │  {y_pred[i]:8.2f}  │    {residuos[i]:7.2f}    │    {residuos_cuad[i]:8.2f}      │')

print('│├─────┼──────────┼──────────┼──────────────┼────────────────┤')
print(f'││ SUM │  {np.sum(y):8.2f}  │  {np.sum(y_pred):8.2f}  │    {np.sum(residuos):7.2f}    │    {np.sum(residuos_cuad):8.2f}      │')
print('│└─────┴──────────┴──────────┴──────────────┴────────────────┘\n')

# Cálculo de MSE y RMSE
MSE = np.sum(residuos_cuad) / n
RMSE = np.sqrt(MSE)

# Cálculo de R²
ss_tot = np.sum((y - y_media) ** 2)
ss_res = np.sum(residuos_cuad)
r_cuadrado = 1 - (ss_res / ss_tot)

print('PASO 2: Calcular MSE, RMSE y R²')
print(f'│ MSE = Σ(eᵢ)² / n = {np.sum(residuos_cuad):.2f} / {n} = {MSE:.4f}')
print(f'│ RMSE = sqrt(MSE) = sqrt({MSE:.4f}) = {RMSE:.4f}')
print(f'│ R² = 1 - (SS_res / SS_tot) = {r_cuadrado:.4f}\n')

print('INTERPRETACION:')
print('┌────────────────────────────────────────────────────────────────────┐')
print(f'│ El modelo comete un error promedio de ±{RMSE:.2f} en precio        │')
print(f'│ El modelo explica el {r_cuadrado*100:.2f}% de la variabilidad      │')
print('└────────────────────────────────────────────────────────────────────┘\n')


# ═════════════════════════════════════════════════════════════════════════
#  GRÁFICAS
# ═════════════════════════════════════════════════════════════════════════

# Gráfica 1: Regresión Lineal
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('BLOQUE 1: Regresión Lineal Simple', fontsize=15, fontweight='bold')

ax1 = axes[0]
ax1.plot(x, y, 'o', markersize=14, linewidth=2.5, color=[0.2, 0.4, 0.8], label='Datos originales')

x_linea = np.linspace(x.min() - 0.2, x.max() + 0.2, 100)
y_linea = a + b * x_linea
ax1.plot(x_linea, y_linea, 'r-', linewidth=3, label='Línea de regresión')

ax1.plot(x, y_pred, 's', markersize=12, linewidth=2, color=[0.8, 0.2, 0.2], label='Predicciones')

for i in range(n):
    ax1.plot([x[i], x[i]], [y[i], y_pred[i]], 'k--', linewidth=1.5)

ax1.set_xlabel('Cantidad de Registros (x)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Precio Promedio (y)', fontsize=12, fontweight='bold')
ax1.set_title('Regresión Lineal Simple', fontsize=13, fontweight='bold')
ax1.text(x.min() + 0.1, y.max() - 1, f'ŷ = {a:.2f} + {b:.4f}·x', fontsize=11, 
         fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='black'))
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')

ax2 = axes[1]
ax2.axis('off')
texto = f'''COMPONENTES DE LA REGRESIÓN

Intersección (a) = {a:.4f}
Pendiente (b) = {b:.4f}

Ecuación: ŷ = a + b·x

MSE = {MSE:.4f}
RMSE = {RMSE:.4f}
R² = {r_cuadrado:.4f}'''

ax2.text(0.1, 0.5, texto, fontsize=11, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=[0.95, 0.95, 1], 
         edgecolor=[0.2, 0.4, 0.8], linewidth=2))

plt.tight_layout()
graph1_path = os.path.join(script_dir, 'bloque1_regresion_lineal.png')
plt.savefig(graph1_path, dpi=100, bbox_inches='tight')
print(f'✓ Gráfica BLOQUE 1 guardada: {graph1_path}')
plt.show()


# Gráfica 2: MSE y RMSE
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('BLOQUE 2: Métricas de Error', fontsize=15, fontweight='bold')

ax1 = axes[0]
bars = ax1.bar(range(1, n+1), residuos_cuad, color=[0.8, 0.2, 0.2], edgecolor='black', linewidth=1.5)
ax1.axhline(y=MSE, color='blue', linestyle='--', linewidth=2.5, label=f'MSE = {MSE:.4f}')
ax1.set_xlabel('Observación (i)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error Cuadrado (eᵢ²)', fontsize=12, fontweight='bold')
ax1.set_title('Errores Cuadrados (Penalización)', fontsize=13, fontweight='bold')
ax1.grid(True, axis='y', alpha=0.3)
ax1.legend()

ax2 = axes[1]
ax2.axis('off')
texto = f'''COMPARACION: MSE vs RMSE

MSE = {MSE:.4f} (en precio²)

RMSE = {RMSE:.4f} (en precio)

R² = {r_cuadrado:.4f}

RMSE es mejor para reportar
en unidades originales'''

ax2.text(0.1, 0.5, texto, fontsize=11, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=[1, 0.95, 0.8],
         edgecolor=[0.8, 0.2, 0.2], linewidth=2))

plt.tight_layout()
graph2_path = os.path.join(script_dir, 'bloque2_mse_rmse.png')
plt.savefig(graph2_path, dpi=100, bbox_inches='tight')
print(f'✓ Gráfica BLOQUE 2 guardada: {graph2_path}')
plt.show()


print('\n╔════════════════════════════════════════════════════════════════════╗')
print('║                    FIN DE LA CLASE                                 ║')
print('╚════════════════════════════════════════════════════════════════════╝\n')

print('✓ Análisis completado exitosamente.\n')
