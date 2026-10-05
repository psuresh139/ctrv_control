"""Regression tests for numerical correctness and reproducible failure scenarios."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi2
from ctrv.model import predict_state_ctrv, jacobian_numerical, jacobian_F_analytical, wrap_to_pi
from ctrv.ekf import ukf_step, ekf_step, compute_nis_summary
from ctrv.noise import calculate_Q
from ctrv.lqr import linearize_at_point
from ctrv.trajectory import build_reference_from_waypoints
from ctrv.limits import DEFAULT_LIMITS, VehicleLimits
from ctrv.simulation import run_simulation, prepare_reference, SensorScenario, path_errors
from ctrv.constants import WAYPOINTS, SEGMENT_COMMANDS, WAYPOINTS_SOFT, SEGMENT_COMMANDS_SOFT


@pytest.mark.parametrize('omega,alpha', [(0., 0.), (1e-9, .5), (.6, -1.), (-.4, .8)])
def test_controlled_motion_matches_continuous_integral(omega, alpha):
    x = np.array([3., -2., 8., 1.2, omega]); dt=.3; a=1.7
    actual=predict_state_ctrv(x, dt, [a, alpha])
    angle=lambda t:x[3]+omega*t+.5*alpha*t*t
    expected_x=x[0]+quad(lambda t:(x[2]+a*t)*np.cos(angle(t)),0,dt,epsabs=1e-12)[0]
    expected_y=x[1]+quad(lambda t:(x[2]+a*t)*np.sin(angle(t)),0,dt,epsabs=1e-12)[0]
    np.testing.assert_allclose(actual[:2], [expected_x,expected_y], atol=1e-10)
    assert abs(wrap_to_pi(actual[3]-angle(dt))) < 1e-12


def test_braking_integrates_only_until_stopped():
    x=predict_state_ctrv(np.array([0.,0.,1.,0.,0.]), 1., [-2.,0.])
    np.testing.assert_allclose(x[:3], [.25,0.,0.], atol=1e-12)


@pytest.mark.parametrize('heading', [np.pi, -np.pi, np.pi-1e-9])
def test_jacobians_across_heading_branch(heading):
    x=np.array([0.,0.,5.,heading,0.])
    F=jacobian_numerical(x,.1,lambda x,dt:predict_state_ctrv(x,dt))
    np.testing.assert_allclose(F, jacobian_F_analytical(x,.1), atol=1e-6)
    A,B=linearize_at_point(x,np.zeros(2),.1)
    assert abs(A[3,3]-1)<1e-6
    assert abs(B[3,1]-.005)<1e-6


def test_ukf_linear_update_includes_process_noise():
    x=np.array([1.,2.,3.,.2,.1]); P=np.diag([.2,.3,.4,.01,.02])
    H=np.eye(5)[:2]; R=np.eye(2)*.7; z=np.array([1.2,1.7])
    Q=calculate_Q(.1, 5., 1., x[3]); prior=P+Q
    S=H@prior@H.T+R; K=np.linalg.solve(S,H@prior).T
    actual,cov,info=ukf_step(x,P,z,.1,R,5.,1.,predict_fn=lambda x,dt,u:x.copy())
    np.testing.assert_allclose(info['innov_cov'],S,atol=1e-10)
    np.testing.assert_allclose(actual,x+K@(z-H@x),atol=1e-10)
    np.testing.assert_allclose(cov,prior-K@S@K.T,atol=1e-10)


def test_ukf_mean_across_heading_branch():
    x=np.array([1.,2.,3.,np.pi-1e-4,.1]); P=np.diag([.2,.3,.4,.01,.02])
    actual,cov,_=ukf_step(x,P,None,.1,np.empty((0,0)),0.,0.,
                         H_meas=np.empty((0,5)),predict_fn=lambda x,dt,u:np.r_[x[:3],wrap_to_pi(x[3]),x[4]])
    assert abs(wrap_to_pi(actual[3]-x[3]))<1e-10
    np.testing.assert_allclose(cov,P,atol=1e-10)


@pytest.mark.parametrize('step', [ukf_step,ekf_step])
def test_predict_only_and_heading_only(step):
    x=np.array([0.,0.,5.,np.pi-.01,.1]); P=np.eye(5)*.01
    kwargs=dict(dt=.1,q_a=.2,q_alpha=.1,predict_fn=predict_state_ctrv)
    prior,cov,info=step(x,P,None,R_meas=np.empty((0,0)),H_meas=np.empty((0,5)),**kwargs)
    assert np.isnan(info['nis']) and not info['gated']
    H=np.eye(5)[3:4]
    posterior,Ppost,info=step(x,P,np.array([-np.pi+.02]),R_meas=np.eye(1)*.01,H_meas=H,**kwargs)
    assert abs(info['innov'][0])<.1
    assert np.trace(Ppost)<np.trace(cov)
    assert np.min(np.linalg.eigvalsh(Ppost))>0


@pytest.mark.parametrize('points,commands', [(WAYPOINTS,SEGMENT_COMMANDS),(WAYPOINTS_SOFT,SEGMENT_COMMANDS_SOFT)])
def test_reference_consistency_and_limits(points,commands):
    ref=build_reference_from_waypoints(points,commands); x=ref['states']; u=ref['u_ref']; L=DEFAULT_LIMITS
    np.testing.assert_allclose(x[[0,-1],:2],points[[0,-1]],atol=1e-10)
    assert np.max(np.linalg.norm(ref['defects'][:,:2],axis=1))<.02
    assert np.max(np.abs(ref['defects'][:,3]))<.002
    np.testing.assert_allclose(ref['defects'][:,[2,4]],0,atol=1e-10)
    np.testing.assert_allclose(x[:,4],x[:,2]*ref['curvature'],atol=1e-10)
    assert np.max(x[:,2])<=L.max_speed+1e-8
    assert np.max(np.abs(x[:,2]*x[:,4]))<=L.max_lateral_accel+1e-8
    assert np.max(np.abs(x[:,4]))<=L.max_yaw_rate+1e-8
    assert np.max(u[:,0])<=L.max_accel+1e-8
    assert np.min(u[:,0])>=-L.max_braking-1e-8
    assert np.max(np.abs(u[:,1]))<=L.max_yaw_accel+1e-8
    assert np.max(np.abs(ref['steering']))<=L.max_steering+1e-8
    assert np.max(np.abs(np.diff(ref['steering'])))/.1<=L.max_steering_rate+1e-8


def test_invalid_and_infeasible_reference_rejected():
    with pytest.raises(ValueError):
        build_reference_from_waypoints(np.array([[0,0],[0,0],[1,1]]),np.ones((2,2)))
    with pytest.raises(ValueError,match='curvature'):
        build_reference_from_waypoints(WAYPOINTS,SEGMENT_COMMANDS,limits=VehicleLimits(max_steering=.001))


def test_nis_keeps_outliers_and_handles_missing_dimensions():
    nis=np.array([1.,100.,np.nan,3.]); dof=np.array([1,2,0,3])
    summary=compute_nis_summary(nis,degrees_of_freedom=dof)
    assert summary['count']==3
    assert summary['mean']==pytest.approx(104/3)
    assert summary['normalized_mean']==pytest.approx(52/3)
    assert summary['pct_above_gate']==pytest.approx(100/3)
    assert compute_nis_summary([np.nan],degrees_of_freedom=[0])['mean'] is None


def test_polyline_distance_and_progress():
    path=np.array([[0,0,0,0,0],[10,0,0,0,0],[20,0,0,0,0]],float)
    distance,progress,heading=path_errors(np.array([[5,2],[20,3]]),path)
    np.testing.assert_allclose(distance,[2,3]);np.testing.assert_allclose(progress,[.25,1.])


@pytest.fixture(scope='module')
def prepared():
    return prepare_reference()


def test_simulation_reproducible_and_does_not_reset_global_rng(prepared):
    np.random.seed(171); state=np.random.get_state()
    a=run_simulation(seed=3,prepared=prepared,debug=False)
    b=run_simulation(seed=3,prepared=prepared,debug=False)
    np.testing.assert_array_equal(a['true_states'],b['true_states'])
    stable=lambda r:{k:v for k,v in r['metrics'].items() if not k.startswith('controller_time_') and k!='deadline_miss_pct'}
    assert stable(a)==stable(b)
    current=np.random.get_state()
    np.testing.assert_array_equal(state[1],current[1]); assert state[2:]==current[2:]


@pytest.mark.parametrize('heading,expected_dof', [(5.,1),(None,0)])
def test_outage_drops_gps_without_fabricating_measurements(prepared,heading,expected_dof):
    r=run_simulation(prepared=prepared,debug=False,heading_sigma_deg=heading,
                     scenario=SensorScenario(gps_outage_duration=5.))
    mask=~r['gps_available']
    assert mask.sum()==50
    assert np.all(r['nis_dof'][mask]==expected_dof)
    if expected_dof==0:
        assert np.all(np.isnan(r['nis_log'][mask]))
    assert np.all(np.isfinite(r['est_states']))
    assert np.all(np.linalg.eigvalsh(r['covariances'])>0)


def test_filter_tuning_does_not_change_true_noise_config(prepared):
    r=run_simulation(prepared=prepared,debug=False,q_a=.1,q_alpha=.01)
    assert r['config']['plant_q_a']==2.
    assert r['config']['plant_q_alpha']==.05
    assert r['config']['filter_q_a']==.1


def test_rejected_innovations_remain_in_summary(prepared):
    r=run_simulation(prepared=prepared,debug=False,
                     scenario=SensorScenario(gps_outlier_probability=1.,gps_outlier_sigma=50.))
    assert np.any(r['gate_log'])
    assert r['nis_summary']['count']==len(r['nis_log'])
    assert r['nis_summary']['mean']==pytest.approx(np.mean(r['nis_log']))


def test_failed_benchmark_runs_count_in_denominator():
    from scripts.run_benchmarks import aggregate
    rows=[dict(scenario='nominal',estimator='ukf',status='failed'),
          dict(scenario='nominal',estimator='ukf',status='ok',completed=True,stayed_on_track=True,success=True)]
    summary=aggregate(rows)[0]
    assert summary['failed_runs']==1 and summary['success_rate']==.5
