"""
triple_pendulum.py
==================
Simulation and real-time visualization of a cart-mounted TRIPLE inverted
pendulum stabilized by the Rho Algorithm (Predictive Convolutional Controller)
combined with a primary cart PID driver.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from RhoController import RhoController

m0 = 0.65                  # Cart mass (kg)
m1, m2, m3 = 0.08, 0.05, 0.02  # Link masses (kg)
L1, L2, L3 = 0.18, 0.14, 0.10  # Link lengths (m)
lc1, lc2, lc3 = L1 / 2.0, L2 / 2.0, L3 / 2.0
I1 = (1.0 / 12.0) * m1 * L1**2
I2 = (1.0 / 12.0) * m2 * L2**2
I3 = (1.0 / 12.0) * m3 * L3**2
g = 9.81                   # Gravity (m/s^2)

# Friction & constraints
b0, b1, b2, b3 = 1.5, 0.002, 0.001, 0.0005
X_LIMIT = 0.25             # Track half-limit (m)
F_MAX = 35.0               # Max motor force (N)

# Simulation sampling: 200 Hz for high-frequency 3rd link whipping mode
DT = 0.005                 # Timestep (s)
TOTAL_TIME = 10.0          # Duration (s)
N_STEPS = int(TOTAL_TIME / DT)  # 2000 steps
DECIMATE = 4               # Render every 4th step (50 FPS display rate)


# Primary Cart PID
KP_CART = -14.14
KI_CART = 0.00
KD_CART = -25.81

# Multichannel Rho / State Feedback Controller
# Channels: [x, theta1, theta2, theta3]
# Alternating sign weights derived from CARE:
# u = -K*s = +14.14*x + 380.54*th1 - 1778.35*th2 + 1602.94*th3 + 25.81*dx - 8.66*dth1 - 33.80*dth2 + 66.75*dth3
W_CHANNELS = [0.0, -1.0, 1.0, -1.0]
KP_RHO = np.array([0.0, 380.54, 1778.35, 1602.94])
KI_RHO = np.array([0.0, 0.0, 0.0, 0.0])
KD_RHO = np.array([0.0, -8.66, 33.80, 66.75])
RHO = 1.0

N_BUFFER = 15
ALPHA_KERN = 0.5
H_PRED = 2
Q_REF = np.zeros(4)  # Target: center, upright

# Lagrangian Equations of Motion

def get_dynamics_matrices(state):
    x, th1, th2, th3, dx, dth1, dth2, dth3 = state
    s1, c1 = np.sin(th1), np.cos(th1)
    s2, c2 = np.sin(th2), np.cos(th2)
    s3, c3 = np.sin(th3), np.cos(th3)

    # Symmetric Mass Matrix M(q)
    M00 = m0 + m1 + m2 + m3
    M01 = (m1 * lc1 + m2 * L1 + m3 * L1) * c1
    M02 = (m2 * lc2 + m3 * L2) * c2
    M03 = m3 * lc3 * c3

    M11 = m1 * lc1**2 + m2 * L1**2 + m3 * L1**2 + I1
    M12 = (m2 * lc2 + m3 * L2) * L1 * np.cos(th1 - th2)
    M13 = m3 * lc3 * L1 * np.cos(th1 - th3)

    M22 = m2 * lc2**2 + m3 * L2**2 + I2
    M23 = m3 * lc3 * L2 * np.cos(th2 - th3)

    M33 = m3 * lc3**2 + I3

    M = np.array([
        [M00, M01, M02, M03],
        [M01, M11, M12, M13],
        [M02, M12, M22, M23],
        [M03, M13, M23, M33],
    ])

    # Gravity vector G(q) = dV/dq
    G = np.array([
        0.0,
        -(m1 * lc1 + m2 * L1 + m3 * L1) * g * s1,
        -(m2 * lc2 + m3 * L2) * g * s2,
        -m3 * lc3 * g * s3,
    ])

    # Coriolis & Centripetal vector C(q, qdot)
    C0 = ((m1 * lc1 + m2 * L1 + m3 * L1) * s1 * dth1**2 +
          (m2 * lc2 + m3 * L2) * s2 * dth2**2 +
          m3 * lc3 * s3 * dth3**2)

    C1 = ((m2 * lc2 + m3 * L2) * L1 * np.sin(th1 - th2) * dth2**2 +
          m3 * lc3 * L1 * np.sin(th1 - th3) * dth3**2)

    C2 = (-(m2 * lc2 + m3 * L2) * L1 * np.sin(th1 - th2) * dth1**2 +
          m3 * lc3 * L2 * np.sin(th2 - th3) * dth3**2)

    C3 = (-m3 * lc3 * L1 * np.sin(th1 - th3) * dth1**2 -
          m3 * lc3 * L2 * np.sin(th2 - th3) * dth2**2)

    C = np.array([C0, C1, C2, C3])

    return M, C, G


def plant_ode(state, u_val):
    x = state[0]
    dx, dth1, dth2, dth3 = state[4:8]

    # Track boundary bumper
    wall_force = 0.0
    if x > X_LIMIT:
        wall_force = -10000.0 * (x - X_LIMIT) - 100.0 * dx
    elif x < -X_LIMIT:
        wall_force = -10000.0 * (x + X_LIMIT) - 100.0 * dx

    u_total = np.clip(u_val, -F_MAX, F_MAX) + wall_force
    friction = np.array([b0 * dx, b1 * dth1, b2 * dth2, b3 * dth3])

    M, C, G = get_dynamics_matrices(state)
    qddot = np.linalg.solve(M, np.array([u_total, 0.0, 0.0, 0.0]) - C - G - friction)

    return np.array([dx, dth1, dth2, dth3, qddot[0], qddot[1], qddot[2], qddot[3]])


def rk4_step(state, u_val):
    k1 = plant_ode(state, u_val)
    k2 = plant_ode(state + 0.5 * DT * k1, u_val)
    k3 = plant_ode(state + 0.5 * DT * k2, u_val)
    k4 = plant_ode(state + DT * k3, u_val)
    return state + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# Initial perturbed angle state
initial_state = np.array([0.0, 0.04, -0.03, 0.02, 0.0, 0.0, 0.0, 0.0])
t_arr = np.linspace(0.0, TOTAL_TIME, N_STEPS + 1)
states = np.zeros((N_STEPS + 1, 8))
u_hist = np.zeros(N_STEPS)
states[0] = initial_state

pid_integral = 0.0
pid_prev_err = 0.0

rho_ctrl = RhoController(
    kp_rho=KP_RHO,
    ki_rho=KI_RHO,
    kd_rho=KD_RHO,
    N=N_BUFFER,
    alpha=ALPHA_KERN,
    H=H_PRED,
    channel_weights=W_CHANNELS,
    dt=DT,
)

print("Simulating 10s Triple Inverted Pendulum with Rho Controller (200 Hz)...")
for i in range(N_STEPS):
    s = states[i]
    q = s[:4]
    qdot = s[4:8]
    e = Q_REF - q

    # Primary Cart PID
    e_x = e[0]
    pid_integral += e_x * DT
    pid_deriv = (e_x - pid_prev_err) / DT
    u_pid = KP_CART * e_x + KI_CART * pid_integral + KD_CART * pid_deriv
    pid_prev_err = e_x

    # Multichannel Rho Controller
    u_rho = rho_ctrl.update(e, qdot=qdot)

    # Actuator force command
    u = np.clip(u_pid + RHO * u_rho, -F_MAX, F_MAX)
    u_hist[i] = u

    # Physics integration
    states[i + 1] = rk4_step(s, u)

    if np.any(np.isnan(states[i + 1])) or np.any(np.abs(states[i + 1]) > 1e4):
        print(f"Divergence detected at {i * DT:.2f}s. Simulation stopped.")
        states[i + 1:] = states[i]
        break

max_ang = np.max(np.abs(states[:, 1:4]), axis=0)
print(f"Completed! Max Deviations -> th1: {np.degrees(max_ang[0]):.1f} deg, th2: {np.degrees(max_ang[1]):.1f} deg, th3: {np.degrees(max_ang[2]):.1f} deg")
print(f"Final State: x = {states[-1, 0]:.4f} m, th1 = {np.degrees(states[-1, 1]):.2f} deg, th2 = {np.degrees(states[-1, 2]):.2f} deg, th3 = {np.degrees(states[-1, 3]):.2f} deg")
print("Launching animation...")

# Decimate for smooth 50 FPS display
anim_indices = np.arange(0, N_STEPS + 1, DECIMATE)
anim_t = t_arr[anim_indices]
anim_states = states[anim_indices]
n_anim_frames = len(anim_indices)


fig, (ax_anim, ax_plot) = plt.subplots(1, 2, figsize=(13, 5))

# Animation View
ax_anim.set_xlim(-0.6, 0.6)
ax_anim.set_ylim(-0.4, 0.6)
ax_anim.set_aspect('equal')
ax_anim.grid(True)
ax_anim.set_title("Triple Inverted Pendulum - Rho Algorithm")
ax_anim.plot([-X_LIMIT, X_LIMIT], [0, 0], color='black', lw=6, label='Track (0.5m)')

pendulum_line, = ax_anim.plot([], [], 'o-', lw=4, color='crimson', markersize=7)
cart_patch = plt.Rectangle((0, 0), 0.10, 0.05, fc='steelblue', ec='black')
ax_anim.add_patch(cart_patch)
time_text = ax_anim.text(0.03, 0.94, '', transform=ax_anim.transAxes, fontsize=10, va='top')
ax_anim.legend(loc='lower left')

# Live Angle Telemetry View
ax_plot.set_xlim(0, TOTAL_TIME)
ax_plot.set_ylim(-15, 15)
ax_plot.set_xlabel('Time (s)')
ax_plot.set_ylabel('Angle (deg)')
ax_plot.set_title('Triple Link Angles Telemetry')
ax_plot.grid(True)
ax_plot.axhline(0, color='gray', lw=0.8, ls='--')
line_th1, = ax_plot.plot([], [], color='crimson', lw=1.8, label=r'$\theta_1$ (Link 1)')
line_th2, = ax_plot.plot([], [], color='royalblue', lw=1.8, label=r'$\theta_2$ (Link 2)')
line_th3, = ax_plot.plot([], [], color='forestgreen', lw=1.8, label=r'$\theta_3$ (Link 3)')
ax_plot.legend(loc='upper right')

def init():
    pendulum_line.set_data([], [])
    cart_patch.set_xy((-0.05, -0.025))
    line_th1.set_data([], [])
    line_th2.set_data([], [])
    line_th3.set_data([], [])
    return pendulum_line, cart_patch, time_text, line_th1, line_th2, line_th3

anim_start_time = [None]

def update_realtime(frame):
    if anim_start_time[0] is None:
        anim_start_time[0] = time.time()

    elapsed = time.time() - anim_start_time[0]
    idx = min(int(np.searchsorted(anim_t, elapsed)), len(anim_t) - 1)

    x, th1, th2, th3 = anim_states[idx, :4]

    p0x, p0y = x, 0.0
    p1x = p0x + L1 * np.sin(th1)
    p1y = p0y + L1 * np.cos(th1)
    p2x = p1x + L2 * np.sin(th2)
    p2y = p1y + L2 * np.cos(th2)
    p3x = p2x + L3 * np.sin(th3)
    p3y = p2y + L3 * np.cos(th3)

    pendulum_line.set_data([p0x, p1x, p2x, p3x], [p0y, p1y, p2y, p3y])
    cart_patch.set_xy((x - 0.05, -0.025))
    time_text.set_text(f't = {anim_t[idx]:.2f} s')

    t_slice = anim_t[:idx + 1]
    line_th1.set_data(t_slice, np.degrees(anim_states[:idx + 1, 1]))
    line_th2.set_data(t_slice, np.degrees(anim_states[:idx + 1, 2]))
    line_th3.set_data(t_slice, np.degrees(anim_states[:idx + 1, 3]))

    return pendulum_line, cart_patch, time_text, line_th1, line_th2, line_th3

ani = animation.FuncAnimation(
    fig, update_realtime, frames=n_anim_frames, init_func=init,
    interval=20, blit=False, repeat=False
)

plt.tight_layout()
plt.show()
