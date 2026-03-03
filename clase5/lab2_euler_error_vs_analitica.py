import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Carpeta de imágenes ---
img_folder = "clase5/imgs"
os.makedirs(img_folder, exist_ok=True)

# --- Archivo de datos y modelo ---
dataFile = "clase5/datos_lab2_freefall.csv"
MODEL = "FREEFALL"

def cargar_datos(archivo):
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"No se encuentra el archivo: {archivo}")

    if archivo.lower().endswith(".csv"):
        T = pd.read_csv(archivo)
    elif archivo.lower().endswith((".xlsx", ".xls")):
        T = pd.read_excel(archivo)
    else:
        raise ValueError("Formato no soportado (usar CSV o XLSX)")

    nombres = T.columns.tolist()
    idx_t = 0
    for i, col in enumerate(nombres):
        if col.lower() == "t":
            idx_t = i
            break

    t = T.iloc[:, idx_t].to_numpy().reshape(-1)
    assert np.all(np.isfinite(t)), "t contiene NaN o Inf"
    assert np.all(np.diff(t) > 0), "t debe ser estrictamente creciente"

    y_meas = None
    measName = ""
    posibles = ["y", "x", "pos", "position", "desplazamiento"]

    for nm in posibles:
        for col in nombres:
            if col.lower() == nm:
                y_meas = T[col].to_numpy().reshape(-1)
                measName = nm
                return t, y_meas, measName

    if len(nombres) >= 2:
        y_meas = T.iloc[:, 1].to_numpy().reshape(-1)
        measName = "col2"

    return t, y_meas, measName

def configurar_modelo(MODEL, t):
    MODEL = MODEL.upper()
    if MODEL == "FREEFALL":
        g = 9.81
        y0 = 0
        v0 = 0
        def f(_, z): return np.array([z[1], -g])
        z0 = np.array([y0, v0])
        z_ana = np.column_stack((y0 + v0*t - 0.5*g*t**2, v0 - g*t))
        stateLabel = ["y [m]", "v [m/s]"]
        measTarget = 0
    elif MODEL == "SPRING":
        m = 1.0
        k = 25.0
        x0 = 0.10
        v0 = 0
        w = np.sqrt(k/m)
        def f(_, z): return np.array([z[1], -(k/m)*z[0]])
        z0 = np.array([x0, v0])
        z_ana = np.column_stack((x0*np.cos(w*t) + (v0/w)*np.sin(w*t),
                                 -x0*w*np.sin(w*t) + v0*np.cos(w*t)))
        stateLabel = ["x [m]", "v [m/s]"]
        measTarget = 0
    else:
        raise ValueError("Modelo no reconocido. Opciones: FREEFALL | SPRING")
    return f, z0, z_ana, stateLabel, measTarget

def euler_solve(f, t, z0):
    N = len(t)
    p = len(z0)
    z = np.zeros((N, p))
    z[0, :] = z0
    for i in range(N-1):
        h = t[i+1] - t[i]
        z[i+1, :] = z[i, :] + h * f(t[i], z[i, :])
    return z

def metricas(a, b):
    r = a - b
    return {"RMSE": np.sqrt(np.mean(r**2)),
            "MAE": np.mean(np.abs(r)),
            "MAX": np.max(np.abs(r))}

def guardar_y_mostrar(fig, nombre):
    fig.savefig(nombre, dpi=150)
    plt.show()

# --- Funciones de graficado ---
def plot_position(t, z_ana, z_num, y_meas=None, zoom_pct=0.1, stateLabel="y [m]", metodo="Euler"):
    N = len(t)
    idx_zoom = int((1 - zoom_pct) * N)
    # Completa
    fig = plt.figure(figsize=(10,5))
    plt.plot(t, z_ana[:,0], label="Analítica", linewidth=2)
    plt.plot(t, z_num[:,0], linestyle='--', label=metodo, linewidth=2)
    if y_meas is not None: plt.plot(t, y_meas, linestyle=':', label="Sensor")
    plt.title(f"Posición - Vista Completa", fontsize=16, fontweight='bold')
    plt.xlabel("Tiempo [s]"); plt.ylabel(stateLabel); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    guardar_y_mostrar(fig, os.path.join(img_folder,"posicion_completa.png"))
    # Zoom
    fig = plt.figure(figsize=(10,5))
    plt.plot(t[idx_zoom:], z_ana[idx_zoom:,0], label="Analítica", linewidth=2)
    plt.plot(t[idx_zoom:], z_num[idx_zoom:,0], linestyle='--', label=metodo, linewidth=2)
    if y_meas is not None: plt.plot(t[idx_zoom:], y_meas[idx_zoom:], linestyle=':', label="Sensor")
    plt.title(f"Posición - Zoom últimos {int(zoom_pct*100)}%", fontsize=16, fontweight='bold')
    plt.xlabel("Tiempo [s]"); plt.ylabel(stateLabel); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    guardar_y_mostrar(fig, os.path.join(img_folder,"posicion_zoom.png"))

def plot_velocity(t, z_ana, z_num, zoom_pct=0.1, stateLabel="v [m/s]", metodo="Euler"):
    N = len(t)
    idx_zoom = int((1 - zoom_pct) * N)
    # Completa
    fig = plt.figure(figsize=(10,5))
    plt.plot(t, z_ana[:,1], label="Analítica", linewidth=2)
    plt.plot(t, z_num[:,1], linestyle='--', label=metodo, linewidth=2)
    plt.title(f"Velocidad - Vista Completa", fontsize=16, fontweight='bold')
    plt.xlabel("Tiempo [s]"); plt.ylabel(stateLabel); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    guardar_y_mostrar(fig, os.path.join(img_folder,"velocidad_completa.png"))
    # Zoom
    fig = plt.figure(figsize=(10,5))
    plt.plot(t[idx_zoom:], z_ana[idx_zoom:,1], label="Analítica", linewidth=2)
    plt.plot(t[idx_zoom:], z_num[idx_zoom:,1], linestyle='--', label=metodo, linewidth=2)
    plt.title(f"Velocidad - Zoom últimos {int(zoom_pct*100)}%", fontsize=16, fontweight='bold')
    plt.xlabel("Tiempo [s]"); plt.ylabel(stateLabel); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    guardar_y_mostrar(fig, os.path.join(img_folder,"velocidad_zoom.png"))

def plot_error(t, error, rmse, max_val, title, zoom_pct=0.1, nombre_archivo_prefix="error"):
    N = len(t)
    idx_zoom = int((1 - zoom_pct) * N)
    # Completo
    fig = plt.figure(figsize=(10,5))
    plt.fill_between(t, 0, error, alpha=0.3, color='red')
    plt.plot(t, error, color='red', linewidth=2)
    plt.axhline(rmse, linestyle='--', color='darkred', label=f"RMSE={rmse:.3e}")
    plt.axhline(max_val, linestyle=':', color='brown', label=f"MAX={max_val:.3e}")
    plt.title(title, fontsize=16, fontweight='bold'); plt.xlabel("Tiempo [s]"); plt.ylabel("|Error|")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    guardar_y_mostrar(fig, os.path.join(img_folder,f"{nombre_archivo_prefix}_completo.png"))
    # Zoom
    fig = plt.figure(figsize=(10,5))
    plt.fill_between(t[idx_zoom:], 0, error[idx_zoom:], alpha=0.5, color='red')
    plt.plot(t[idx_zoom:], error[idx_zoom:], color='red', linewidth=2)
    plt.axhline(rmse, linestyle='--', color='darkred', label=f"RMSE={rmse:.3e}")
    plt.axhline(max_val, linestyle=':', color='brown', label=f"MAX={max_val:.3e}")
    plt.title(f"{title} - Zoom últimos {int(zoom_pct*100)}%", fontsize=16, fontweight='bold')
    plt.xlabel("Tiempo [s]"); plt.ylabel("|Error|")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    guardar_y_mostrar(fig, os.path.join(img_folder,f"{nombre_archivo_prefix}_zoom.png"))

def plot_sensor_vs_model(t, y_meas, z_ana, z_num, measTarget, stateLabel, zoom_pct=0.1):
    if y_meas is None or len(y_meas)==0:
        print("No hay datos de sensor para validación."); return
    N = len(t); idx_zoom = int((1-zoom_pct)*N)
    ev = np.abs(y_meas - z_ana[:,measTarget])
    en = np.abs(y_meas - z_num[:,measTarget])
    rmse_s_ana = np.sqrt(np.mean(ev**2)); rmse_s_num = np.sqrt(np.mean(en**2))
    # Completo
    fig = plt.figure(figsize=(10,5))
    plt.fill_between(t, 0, ev, color='green', alpha=0.25)
    plt.fill_between(t, 0, en, color='red', alpha=0.25)
    plt.plot(t, ev, color='green', linewidth=2, label=f"Sensor vs Analítica (RMSE={rmse_s_ana:.3e})")
    plt.plot(t, en, color='red', linestyle='--', linewidth=2, label=f"Sensor vs Euler (RMSE={rmse_s_num:.3e})")
    plt.axhline(rmse_s_ana, color='green', linestyle='--', linewidth=1.3)
    plt.axhline(rmse_s_num, color='red', linestyle='--', linewidth=1.3)
    plt.title("VALIDACIÓN: Sensor vs Modelo", fontsize=16, fontweight='bold')
    plt.xlabel("Tiempo [s]"); plt.ylabel(f"|Error| {stateLabel}"); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    guardar_y_mostrar(fig, os.path.join(img_folder,"validacion_sensor_modelo_completo.png"))
    # Zoom
    fig = plt.figure(figsize=(10,5))
    plt.fill_between(t[idx_zoom:],0,ev[idx_zoom:],color='green',alpha=0.4)
    plt.fill_between(t[idx_zoom:],0,en[idx_zoom:],color='red',alpha=0.4)
    plt.plot(t[idx_zoom:],ev[idx_zoom:],color='green',linewidth=2,label=f"Sensor vs Analítica (RMSE={rmse_s_ana:.3e})")
    plt.plot(t[idx_zoom:],en[idx_zoom:],color='red',linestyle='--',linewidth=2,label=f"Sensor vs Euler (RMSE={rmse_s_num:.3e})")
    plt.axhline(rmse_s_ana,color='green',linestyle='--',linewidth=1.3)
    plt.axhline(rmse_s_num,color='red',linestyle='--',linewidth=1.3)
    # Anotación máximo error
    max_idx = idx_zoom + np.argmax(ev[idx_zoom:])
    plt.annotate(f"Máx error\n{ev[max_idx]:.3f}", xy=(t[max_idx], ev[max_idx]),
                 xytext=(t[max_idx]-0.3, ev[max_idx]+0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))
    plt.title(f"VALIDACIÓN: Sensor vs Modelo - Zoom últimos {int(zoom_pct*100)}%", fontsize=16, fontweight='bold')
    plt.xlabel("Tiempo [s]"); plt.ylabel(f"|Error| {stateLabel}"); plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    guardar_y_mostrar(fig, os.path.join(img_folder,"validacion_sensor_modelo_zoom.png"))

# --- MAIN ---
if __name__ == "__main__":
    t, y_meas, measName = cargar_datos(dataFile)
    print(f"Datos: {len(t)} pts | t=[{t[0]:.3f}, {t[-1]:.3f}] s | dt={np.median(np.diff(t)):.4g} s")
    f, z0, z_ana, stateLabel, measTarget = configurar_modelo(MODEL, t)
    z_num = euler_solve(f, t, z0)
    # Métricas
    e_pos = np.abs(z_num[:,0]-z_ana[:,0]); m_pos = metricas(z_num[:,0], z_ana[:,0])
    e_vel = np.abs(z_num[:,1]-z_ana[:,1]); m_vel = metricas(z_num[:,1], z_ana[:,1])
    zoom_percentage = 0.1
    # Graficar
    plot_position(t, z_ana, z_num, y_meas, zoom_pct=zoom_percentage, stateLabel=stateLabel[0], metodo="Euler")
    plot_velocity(t, z_ana, z_num, zoom_pct=zoom_percentage, stateLabel=stateLabel[1], metodo="Euler")
    plot_error(t, e_pos, m_pos["RMSE"], m_pos["MAX"], "Error en Posición", zoom_pct=zoom_percentage, nombre_archivo_prefix="error_posicion")
    plot_error(t, e_vel, m_vel["RMSE"], m_vel["MAX"], "Error en Velocidad", zoom_pct=zoom_percentage, nombre_archivo_prefix="error_velocidad")
    plot_sensor_vs_model(t, y_meas, z_ana, z_num, measTarget, stateLabel[measTarget], zoom_pct=zoom_percentage)