"""Durability and pairing contracts for the new vehicle experiment runner."""
import json
import numpy as np
import pytest
from scripts import run_vehicle_benchmarks as bench


def test_seed_pairing_uses_matching_seeds_and_retains_failed_completion():
    rows=[]
    for seed in [0,1,2]:
        rows.append(dict(scenario='nominal',estimator='ekf',controller='mpc',seed=seed,status='ok',
                         completed=True,success=False,cross_track_rmse_m=seed+2.,physical_limit_violation_pct=20.,
                         completion_time_s=40.,solver_fallback_pct=0.))
        if seed==2:
            rows.append(dict(scenario='nominal',estimator='ekf',controller='uncertainty',seed=seed,status='failed'))
        else:
            rows.append(dict(scenario='nominal',estimator='ekf',controller='uncertainty',seed=seed,status='ok',
                             completed=True,success=False,cross_track_rmse_m=seed+1.,physical_limit_violation_pct=10.,
                             completion_time_s=45.,solver_fallback_pct=0.))
    result=bench.paired_ablations(list(reversed(rows)))[0]
    assert result['matched_seeds']==3
    assert result['cross_track_rmse_m_valid_pairs']==2
    assert result['cross_track_rmse_m_difference_mean']==-1.
    assert result['completed_valid_pairs']==3
    assert result['completed_difference_mean']==pytest.approx(-1/3)
    assert result['completion_time_s_valid_pairs']==2
    assert result['completion_time_s_difference_mean']==5.


def test_atomic_checkpoint_resume_and_fingerprint_rejection(tmp_path,monkeypatch):
    count=[]
    monkeypatch.setitem(bench._PREPARED,.9,(None,None))
    def fake(**kwargs):
        count.append(kwargs['seed'])
        return {'metrics':{'completed':False,'cross_track_rmse_m':1.},'config':{'seed':kwargs['seed']},
                'controller_diagnostics':[{'status':'solved'}]}
    monkeypatch.setattr(bench,'run_vehicle_simulation',fake)
    job=('nominal','ekf','mpc',[4],5.,1.,str(tmp_path),False,'fingerprint')
    first=bench.run_group(job);second=bench.run_group(job)
    assert first==second and count==[4]
    assert len(list((tmp_path/'checkpoints').glob('*.json')))==1
    assert not list(tmp_path.rglob('*.tmp'))
    with pytest.raises(ValueError,match='fingerprint'):
        bench.run_group(job[:-1]+('different',))


def test_execution_failure_has_a_checkpoint_and_remains_in_denominator(tmp_path,monkeypatch):
    monkeypatch.setitem(bench._PREPARED,.9,(None,None))
    def fail(**kwargs):raise RuntimeError('injected physical integration error')
    monkeypatch.setattr(bench,'run_vehicle_simulation',fail)
    job=('nominal','ekf','mpc',[0,1],5.,1.,str(tmp_path),False,'fingerprint')
    rows=bench.run_group(job)
    assert len(rows)==2 and all(r['status']=='failed' for r in rows)
    summary=bench.aggregate(rows)[0]
    assert summary['runs']==2 and summary['failed_runs']==2 and summary['completed_rate']==0.
    assert len(list((tmp_path/'checkpoints').glob('*.json')))==2


def test_trace_export_is_pickle_free_and_resume_requires_trace(tmp_path,monkeypatch):
    monkeypatch.setitem(bench._PREPARED,.9,(None,None))
    keys=['iterations','primal_residual','dual_residual','fallback','slack_max','slack_mean','tightening_max','computation_s']
    def fake(**kwargs):
        result={k:np.zeros((1,)) for k in ['times','true_states','estimated_states','covariances','commands','applied_commands','speed_caps','estimated_progress']}
        result.update(metrics={'completed':False},config={},controller_diagnostics=[dict(status='solved',tightening_by_quantity=[0.]*8,**dict.fromkeys(keys,0.))],
                      vehicle_diagnostics=[{'friction_mu':.9,'tire_utilization_max':.2}])
        return result
    monkeypatch.setattr(bench,'run_vehicle_simulation',fake)
    job=('nominal','ekf','mpc',[0],5.,1.,str(tmp_path),True,'fingerprint')
    bench.run_group(job)
    trace=next((tmp_path/'traces').glob('*.npz'))
    with np.load(trace,allow_pickle=False) as data:
        assert data['tightening_by_quantity'].shape==(1,8)
    with np.load(trace,allow_pickle=False) as data:
        assert data['commands'].shape==(1,) and 'tightening_max' in data.files
    trace.unlink()
    with pytest.raises(ValueError,match='trace missing'):bench.run_group(job)
