"""MPC numerical, fallback, replay and benchmark-export acceptance tests."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from ctrv.constants import DT
from ctrv.limits import DEFAULT_LIMITS
from ctrv.model import wrap_to_pi
from ctrv.mpc import MPCConfig,MPCController
from ctrv.simulation import prepare_reference,run_simulation,SCENARIOS


@pytest.fixture(scope='module')
def prepared():return prepare_reference()


def controller(prepared,config=None):
    r,A,B=prepared;return MPCController(r,A,B,config)


def test_config_validation():
    with pytest.raises(ValueError):MPCConfig(horizon=0)
    with pytest.raises(ValueError):MPCConfig(slack_quadratic=-1)


def test_affine_dynamics_rows_match_reference_defects(prepared):
    m=controller(prepared);N,nx,nu,P,C,l,u,q,D,W=m.template(3)
    r,A,B=prepared
    np.testing.assert_allclose(l[5:5+5*N].reshape(N,5),r['defects'][3:3+N],atol=1e-12)
    # Dynamics rows applied to arbitrary deviations equal e_next-Ae-Bdu.
    rng=np.random.default_rng(5);z=rng.normal(size=C.shape[1]);actual=C[5:5+5*N]@z
    expected=[]
    for i in range(N):expected.extend(z[5*(i+1):5*(i+2)]-A[3+i]@z[5*i:5*(i+1)]-B[3+i]@z[nx+2*i:nx+2*i+2])
    np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_hard_actuators_and_warm_start(prepared):
    m=controller(prepared);r=prepared[0];x=r['states'][0]+[2,2,.5,.1,.05]
    u,d=m.command(0,x,r['u_ref'][0],np.zeros(2))
    assert d['solved'] and not d['fallback'] and not d['warm_start']
    assert -5<=u[0]<=3 and -1.5<=u[1]<=1.5
    _,second=m.command(1,r['states'][1],u,np.zeros(2))
    assert second['warm_start'] and second['solved']


def test_heading_chart_equivalent_modulo_two_pi(prepared):
    r=prepared[0];x=r['states'][20].copy();x[3]=np.pi
    a,da=controller(prepared).command(20,x,r['u_ref'][19],np.zeros(2))
    x[3]-=2*np.pi
    b,db=controller(prepared).command(20,x,r['u_ref'][19],np.zeros(2))
    np.testing.assert_allclose(a,b,atol=1e-8)
    assert da['status']==db['status']


def test_slack_activates_for_outside_corridor(prepared):
    r=prepared[0];m=controller(prepared);x=r['states'][0].copy();x[1]+=8.
    _,d=m.command(0,x,r['u_ref'][0],np.zeros(2))
    assert d['solved'] and d['slack_max']>.1


def test_horizon_truncates(prepared):
    r=prepared[0];k=len(r['states'])-2
    _,d=controller(prepared).command(k,r['states'][k],r['u_ref'][k-1],np.zeros(2))
    assert d['horizon']==1 and d['solved']


def test_unsolved_and_exception_fallback(prepared):
    r=prepared[0];m=controller(prepared);state=r['states'][0]
    m.command(0,state,np.zeros(2),np.zeros(2))
    class Fake:
        def update(self,**kw):pass
        def warm_start(self,**kw):pass
        def solve(self,**kw):
            return SimpleNamespace(x=None,info=SimpleNamespace(status='primal infeasible',status_val=3,
                iter=50,prim_res=1.,dual_res=2.))
    m.workspaces[20]=Fake()
    actual,d=m.command(0,state,np.zeros(2),np.array([-8.,4.]))
    np.testing.assert_array_equal(actual,[-5.,1.5]);assert d['fallback']
    class Broken(Fake):
        def solve(self,**kw):raise RuntimeError('injected solver failure')
    m.workspaces[20]=Broken()
    _,d=m.command(0,state,np.zeros(2),np.zeros(2));assert d['fallback'] and 'solver error' in d['status']


def test_missing_osqp_is_install_error(prepared,monkeypatch):
    import builtins
    original=builtins.__import__
    def missing(name,*args,**kw):
        if name=='osqp':raise ImportError('not installed')
        return original(name,*args,**kw)
    monkeypatch.setattr(builtins,'__import__',missing)
    with pytest.raises(ImportError,match='requires OSQP'):controller(prepared)


@pytest.mark.parametrize('estimator',['ekf','ukf'])
def test_deterministic_recovery_no_fallback(prepared,estimator):
    r=run_simulation(prepared=prepared,controller='mpc',estimator=estimator,
                     scenario=SCENARIOS['deterministic_recovery'],debug=False)
    assert r['metrics']['solver_fallback_pct']==0
    assert r['metrics']['completed'] and r['metrics']['cross_track_rmse_m']<.5
    assert np.all(np.isfinite(r['true_states']))


def test_replay_and_seed_pairing(prepared):
    kwargs=dict(prepared=prepared,controller='mpc',scenario=SCENARIOS['deterministic_recovery'],debug=False)
    a=run_simulation(**kwargs);b=run_simulation(**kwargs)
    np.testing.assert_array_equal(a['true_states'],b['true_states'])
    np.testing.assert_array_equal(a['controls'],b['controls'])
    lqr=run_simulation(prepared=prepared,debug=False)
    assert a['config']['plant_q_a']==lqr['config']['plant_q_a']
    assert a['config']['initial_estimate_offset']==lqr['config']['initial_estimate_offset']


def test_paired_export_and_failed_denominators():
    from scripts.run_benchmarks import paired_comparisons,aggregate
    rows=[]
    for seed in range(3):
        for ctl in ['lqr','mpc']:
            rows.append(dict(scenario='nominal',estimator='ukf',controller=ctl,seed=seed,status='ok',
                         cross_track_rmse_m=2. if ctl=='lqr' else 1.,physical_limit_violation_pct=10.,
                         completed=True,success=False))
    pairs=paired_comparisons(rows)[0]
    assert pairs['cross_track_rmse_m_difference_mean']==-1
    assert pairs['cross_track_rmse_m_ci95']==[-1,-1]
    assert len(aggregate(rows))==2
    json.dumps(pairs,allow_nan=False)


def test_atomic_checkpoint_and_fingerprint(tmp_path):
    from scripts.run_benchmarks import atomic_json,fingerprint
    path=tmp_path/'checkpoints'/'run.json';atomic_json(path,{'finite':1.})
    assert json.loads(path.read_text())=={'finite':1.} and not path.with_suffix('.json.tmp').exists()
    env=dict(source_sha256='a',python='3.12',platform='test',dependencies={'osqp':'1'})
    assert fingerprint({'seed':1},env)!=fingerprint({'seed':2},env)
    other=dict(env,source_sha256='b');assert fingerprint({},env)!=fingerprint({},other)


def test_cli_exports_and_resume(tmp_path):
    from scripts.run_benchmarks import main
    args=['--seeds','1','--scenarios','deterministic_recovery','--controllers','lqr','--estimators','ukf',
          '--save-traces','--output',str(tmp_path)]
    assert main(args)==0
    runs=json.loads((tmp_path/'runs.json').read_text());assert len(runs)==1 and runs[0]['status']=='ok'
    path=next((tmp_path/'checkpoints').glob('*.json'));before=path.stat().st_mtime_ns
    assert main(args+['--resume'])==0 and path.stat().st_mtime_ns==before
    with pytest.raises(SystemExit):main(args+['--resume','--seeds','2'])
    with np.load(next(tmp_path.glob('*.npz'))) as trace:
        assert 'solver_status' in trace and 'controller_time_s' in trace
    assert json.loads((tmp_path/'manifest.json').read_text())['state']=='complete'


def test_actual_cli_interfaces(tmp_path):
    import os,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR=str(tmp_path/'cache'),PYTHONDONTWRITEBYTECODE='1')
    plot=tmp_path/'mpc.png'
    result=subprocess.run([sys.executable,'scripts/run_simulation.py','--controller','mpc',
                           '--scenario','deterministic_recovery','--save',str(plot)],
                           cwd=root,env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert plot.read_bytes().startswith(b'\x89PNG')
    result=subprocess.run([sys.executable,'scripts/run_benchmarks.py','--help'],
                           cwd=root,env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0 and '--controllers' in result.stdout and '--resume' in result.stdout
