"""
ctrv — CTRV vehicle estimation and control package.

Public API
----------
from ctrv.model      import predict_state_ctrv, jacobian_F_analytical, wrap_to_pi
from ctrv.noise      import calculate_Q
from ctrv.ekf        import ukf_step, ekf_step, UKFParams,
                            compute_nis_summary, print_nis_report
from ctrv.lqr        import (build_ltv_linearization, ltv_lqr_backward_riccati,
                              pick_k_nearest, compute_lqr_gain_dare)
from ctrv.trajectory import build_reference_from_waypoints
from ctrv.constants  import *   # state indices, dimensions, defaults
"""

from ctrv.constants  import (
    N_STATES, N_MEAS, N_CONTROL,
    IDX_X, IDX_Y, IDX_V, IDX_PSI, IDX_PSIDOT,
    EPS, DT,
    Q_A_PSD, Q_ALPHA_PSD, R_MEAS, P_INIT, H_MEAS,
    H_MEAS_2DOF, H_MEAS_3DOF, R_MEAS_3DOF,
    WAYPOINTS, SEGMENT_COMMANDS, U_BOUNDS,
    WAYPOINTS_SOFT, SEGMENT_COMMANDS_SOFT,
    Q_LQR, R_LQR, INNOVATION_GATE,
)
from ctrv.model      import (
    wrap_to_pi,
    predict_state_ctrv,
    predict_state_ctrv_baseline,
    jacobian_F_analytical,
    jacobian_numerical,
)
from ctrv.noise      import calculate_Q
from ctrv.ekf        import (
    ukf_step, ekf_step, UKFParams,
    compute_nis_summary, print_nis_report,
)
from ctrv.lqr        import (
    linearize_at_point,
    build_ltv_linearization,
    ltv_lqr_backward_riccati,
    ltv_lqr_scheduled_riccati,
    compute_lqr_gain_dare,
    pick_k_nearest,
)
from ctrv.trajectory import build_reference_from_waypoints
