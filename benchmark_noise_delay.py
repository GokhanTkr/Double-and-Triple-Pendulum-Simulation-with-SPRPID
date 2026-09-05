"""
benchmark_noise_delay.py
========================
Rigorous comparative benchmark under realistic hardware constraints:
  - Sensor Quantization Noise: sigma = 0.002 rad (~0.11 degrees)
  - Communication / Sensing Delay: 10 ms (2 sample steps at 200 Hz)

Compares:
  1. Standard Low-Pass Filtered LQR (LPF introduces phase lag -> instability)
  2. Naive Finite Difference LQR (no filter -> extreme motor chatter & torque spikes)
  3. Rho Predictive Convolution (FIR filtering + H-step predictive phase lead)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are


m0 = 0.65; m1, m2 = 0.08, 0.05
L1, L2 = 0.18, 0.14
lc1, lc2 = L1 / 2.0, L2 / 2.0
I1 = (1.0 / 12.0) * m1 * L1**2
I2 = (1.0 / 12.0) * m2 * L2**2
g = 9.81
b0, b1, b2 = 1.5, 0.002, 0.001

DT = 0.005                   # 200 Hz
TOTAL_TIME = 5.0             # 5 seconds
N_STEPS = int(TOTAL_TIME / DT)
F_MAX = 25.0

# Linearized mass matrix at upright
M0 = np.array([
    [m0 + m1 + m2, m1*lc1 + m2*L1, m2*lc2],
    [m1*lc1 + m2*L1, m1*lc1**2 + m2*L1**2 + I1, m2*lc2*L1],
    [m2*lc2, m2*lc2*L1, m2*lc2**2 + I2],
])
Minv = np.linalg.inv(M0)
b_vec = Minv[:, 0]
g1 = (m1 * lc1 + m2 * L1) * g
g2 = m2 * lc2 * g

A = np.zeros((6, 6))
A[0, 3] = 1.0; A[1, 4] = 1.0; A[2, 5] = 1.0
A[3, 1] = Minv[0, 1] * g1; A[3, 2] = Minv[0, 2] * g2; A[3, 3] = -Minv[0, 0] * b0
A[4, 1] = Minv[1, 1] * g1; A[4, 2] = Minv[1, 2] * g2; A[4, 4] = -Minv[1, 1] * b1
A[5, 1] = Minv[2, 1] * g1; A[5, 2] = Minv[2, 2] * g2; A[5, 5] = -Minv[2, 2] * b2

B = np.zeros((6, 1))
B[3:, 0] = b_vec

Q = np.diag([40.0, 200.0, 150.0, 5.0, 10.0, 10.0])
R = np.array([[0.02]])
P = solve_continuous_are(A, B, Q, R)
K = (np.linalg.inv(R) @ B.T @ P).flatten()


def dyn(s):
    x, th1, th2, dx, dth1, dth2 = s
    s1, c1 = np.sin(th1), np.cos(th1)
    s2, c2 = np.sin(th2), np.cos(th2)

    M = np.array([
        [m0 + m1 + m2, (m1*lc1 + m2*L1)*c1, m2*lc2*c2],
        [(m1*lc1 + m2*L1)*c1, m1*lc1**2 + m2*L1**2 + I1, m2*lc2*L1*np.cos(th1 - th2)],
        [m2*lc2, m2*lc2*L1*np.cos(th1 - th2), m2*lc2**2 + I2],
    ])
    G = np.array([0.0, -(m1*lc1 + m2*L1)*g*s1, -m2*lc2*g*s2])
    C = np.array([
        (m1*lc1 + m2*L1)*s1*dth1**2 + m2*lc2*s2*dth2**2,
        m2*lc2*L1*np.sin(th1 - th2)*dth2**2,
        -m2*lc2*L1*np.sin(th1 - th2)*dth1**2,
    ])
    return M, C, G


def ode(s, u):
    M, C, G = dyn(s)
    fr = np.array([b0 * s[3], b1 * s[4], b2 * s[5]])
    qdd = np.linalg.solve(M, np.array([np.clip(u, -F_MAX, F_MAX), 0.0, 0.0]) - C - G - fr)
    return np.array([s[3], s[4], s[5], qdd[0], qdd[1], qdd[2]])


def rk4(s, u):
    k1 = ode(s, u)
    k2 = ode(s + 0.5 * DT * k1, u)
    k3 = ode(s + 0.5 * DT * k2, u)
    k4 = ode(s + DT * k3, u)
    return s + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


DELAY_STEPS = 2            # 2 steps * 5 ms = 10 ms transport delay
NOISE_SIGMA = 0.002        # ~0.11 degrees encoder quantization noise

def run_simulation(controller_mode):
    np.random.seed(123)    # identical noise sequence for fair comparison
    s = np.array([0.0, 0.04, -0.02, 0.0, 0.0, 0.0])
    
    t_history = np.linspace(0.0, TOTAL_TIME, N_STEPS + 1)
    states_hist = np.zeros((N_STEPS + 1, 6))
    u_hist = np.zeros(N_STEPS)
    states_hist[0] = s.copy()

    # Hardware delay buffer
    delay_buffer = [s[:3].copy() for _ in range(DELAY_STEPS + 1)]
    q_prev = s[:3].copy()
    
    # Filter state registers
    lpf_vel = np.zeros(3)
    alpha_lpf = 0.25       # 1st-order IIR low-pass filter coefficient
    
    # Rho FIR buffer
    N_BUF = 15
    alpha_fir = 0.5
    k_fir = np.exp(-alpha_fir * np.arange(N_BUF))
    k_fir /= k_fir.sum()
    meas_history = []
    conv_history = []

    for i in range(N_STEPS):
        # 1. Noisy sensor measurement (Position-only)
        noise = np.random.normal(0, NOISE_SIGMA, 3)
        q_raw = s[:3] + noise

        # 2. Hardware delay queue (measurement arrives DELAY_STEPS later)
        delay_buffer.append(q_raw.copy())
        q_delayed = delay_buffer.pop(0)

        # 3. Velocity estimation per method
        if controller_mode == "Standard Filtered LQR":
            # Standard engineering approach: raw diff + low pass filter
            v_raw = (q_delayed - q_prev) / DT
            lpf_vel = (1.0 - alpha_lpf) * lpf_vel + alpha_lpf * v_raw
            qdot_est = lpf_vel.copy()
            q_est = q_delayed.copy()

        elif controller_mode == "Naive Finite Diff LQR":
            # No filter: raw finite differences (fast, but noise explodes)
            qdot_est = (q_delayed - q_prev) / DT
            q_est = q_delayed.copy()

        elif controller_mode == "Rho Predictive Convolution":
            # Rho method: FIR convolution on history + H-step predictive lead
            meas_history.append(q_delayed.copy())
            if len(meas_history) > N_BUF:
                meas_history.pop(0)

            stack = np.stack(meas_history[::-1])
            n = len(stack)
            k_sub = k_fir[:n] / k_fir[:n].sum()
            q_conv = k_sub @ stack
            conv_history.append(q_conv)

            # Predictive derivative with curvature compensation for transport delay
            if len(conv_history) >= 3:
                v1 = (conv_history[-1] - conv_history[-2]) / DT
                v0 = (conv_history[-2] - conv_history[-3]) / DT
                # Extrapolate forward across delay horizon:
                qdot_est = v1 + 1.0 * (v1 - v0)
                q_est = q_conv + (DELAY_STEPS * DT) * qdot_est
            elif len(conv_history) >= 2:
                qdot_est = (conv_history[-1] - conv_history[-2]) / DT
                q_est = q_conv
            else:
                qdot_est = np.zeros(3)
                q_est = q_delayed

        q_prev = q_delayed.copy()

        # 4. State feedback control
        s_feedback = np.hstack([q_est, qdot_est])
        u = -float(np.dot(K, s_feedback))
        u_clipped = np.clip(u, -F_MAX, F_MAX)
        u_hist[i] = u_clipped

        # 5. Physics update
        s = rk4(s, u_clipped)
        states_hist[i + 1] = s.copy()

        if np.any(np.abs(s[1:3]) > 0.8) or np.any(np.isnan(s)):
            states_hist[i + 1:] = s.copy()
            u_hist[i:] = 0.0
            break

    return t_history, states_hist, u_hist


modes = [
    ("Standard Filtered LQR", "crimson", "--"),
    ("Naive Finite Diff LQR", "orange", ":"),
    ("Rho Predictive Convolution", "royalblue", "-"),
]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

print("Running comparative benchmark under 10ms delay + sensor noise...\n")

for mode_name, color, line_style in modes:
    t_h, states_h, u_h = run_simulation(mode_name)
    
    th1_deg = np.degrees(states_h[:, 1])
    th2_deg = np.degrees(states_h[:, 2])
    
    # Calculate performance metrics
    survived_time = t_h[np.where(np.abs(states_h[:, 1]) > 0.8)[0][0]] if np.any(np.abs(states_h[:, 1]) > 0.8) else TOTAL_TIME
    chatter_metric = np.mean(np.abs(np.diff(u_h))) / DT
    
    label_str = f"{mode_name} (Stable: {survived_time:.1f}s)" if survived_time == TOTAL_TIME else f"{mode_name} (FELL at {survived_time:.2f}s)"
    print(f"{mode_name:30s} -> Survived: {survived_time:4.2f}s | Actuator Chatter: {chatter_metric:8.1f} N/s")

    # Plot Link 1 Angle
    ax1.plot(t_h, th1_deg, color=color, linestyle=line_style, lw=2.0, label=label_str)
    # Plot Link 2 Angle
    ax2.plot(t_h, th2_deg, color=color, linestyle=line_style, lw=2.0, label=label_str)
    # Plot Actuator Force
    ax3.plot(t_h[:-1], u_h, color=color, linestyle=line_style, lw=1.5, alpha=0.85)

# Formatting
ax1.set_ylabel(r"$\theta_1$ Link 1 Angle ($^\circ$)", fontsize=11)
ax1.set_title("Performance Comparison Under Real-World Conditions (10ms Delay + Sensor Noise)", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.5)
ax1.legend(loc="upper right", fontsize=9)
ax1.set_ylim(-25, 25)

ax2.set_ylabel(r"$\theta_2$ Link 2 Angle ($^\circ$)", fontsize=11)
ax2.grid(True, alpha=0.5)
ax2.set_ylim(-25, 25)

ax3.set_ylabel("Motor Force $u$ (N)", fontsize=11)
ax3.set_xlabel("Time (s)", fontsize=11)
ax3.grid(True, alpha=0.5)
ax3.set_ylim(-28, 28)

plt.tight_layout()
output_path = r"C:\Users\Tekir\Desktop\pendulum\benchmark_noise_delay.png"
plt.savefig(output_path, dpi=200)
print(f"\nPlot saved to: {output_path}")
plt.show()
