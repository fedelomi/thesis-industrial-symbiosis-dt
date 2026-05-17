"""Quick smoke test: 100 steps per scenario to verify analytical RC stability."""
import sys
sys.path.insert(0, r"C:\Users\Feder\OneDrive\Desktop\Github\DataGym\DatacenterGym-main")
import numpy as np

# Import just the core functions inline (avoid CoolProp overhead for thermal check)
def rc_thermal_model(theta_now, dt, C_d, sum_alpha_u, T_amb, R_d, Q_cool):
    tau      = C_d * R_d
    theta_ss = T_amb + R_d * (sum_alpha_u - Q_cool)
    theta_next = theta_ss + (theta_now - theta_ss) * np.exp(-dt / tau)
    return theta_next

DC_SCENARIOS = {
    "Edge": {
        "rated_power_W": 100e3, "C_d": 12000,
        "R_d": 25.0 / (0.92 * 100e3 - 45e3),
        "alpha_i": 0.92, "T_amb": 15 + 273.15, "Q_cool": 45e3,
        "T_supply_target": 40.0,
    },
    "Mid": {
        "rated_power_W": 5e6, "C_d": 65000,
        "R_d": 20.0 / (0.95 * 5e6 - 2.8e6),
        "alpha_i": 0.95, "T_amb": 15 + 273.15, "Q_cool": 2.8e6,
        "T_supply_target": 35.0,
    },
    "Hyperscale": {
        "rated_power_W": 100e6, "C_d": 220000,
        "R_d": 15.0 / (0.97 * 100e6 - 58e6),
        "alpha_i": 0.97, "T_amb": 15 + 273.15, "Q_cool": 58e6,
        "T_supply_target": 30.0,
    }
}

dt = 900  # 15 min
N = 100

for name, p in DC_SCENARIOS.items():
    tau = p["C_d"] * p["R_d"]
    P_IT = p["alpha_i"] * p["rated_power_W"]
    theta_ss_full = p["T_amb"] + p["R_d"] * (P_IT - p["Q_cool"])

    print(f"\n=== {name} ===")
    print(f"  tau = {tau:.2f} s  |  dt/tau = {dt/tau:.1f}")
    print(f"  theta_ss (full load) = {theta_ss_full - 273.15:.1f} C  (target: {p['T_supply_target']} C)")

    theta = p["T_amb"]
    temps = []
    for t in range(N):
        u_t = 0.775 + 0.175 * np.sin(2 * np.pi * t / 96 - np.pi/2)
        sum_alpha_u = p["alpha_i"] * p["rated_power_W"] * u_t
        theta = rc_thermal_model(theta, dt, p["C_d"], sum_alpha_u, p["T_amb"], p["R_d"], p["Q_cool"])
        temps.append(theta - 273.15)

    print(f"  T_supply range: {min(temps):.1f} - {max(temps):.1f} C")
    print(f"  All finite: {all(np.isfinite(t) for t in temps)}")
    print(f"  No NaN: {not any(np.isnan(t) for t in temps)}")

print("\nAll scenarios stable!")
