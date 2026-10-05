"""Sparse LTV MPC. Soft predicted envelopes are not nonlinear safety guarantees."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy import sparse
from ctrv.constants import Q_LQR, R_LQR, DT
from ctrv.limits import DEFAULT_LIMITS
from ctrv.model import predict_state_ctrv, wrap_to_pi


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 20
    terminal_multiplier: float = 10.
    rate_weights: tuple = (.5, 5.)
    slack_quadratic: float = 10000.
    slack_linear: float = 1000.
    eps_abs: float = 1e-5
    eps_rel: float = 1e-5
    max_iter: int = 4000
    actuator_tolerance: float = 1e-4
    rho: float = 3.0

    def __post_init__(self):
        object.__setattr__(self,'rate_weights',tuple(self.rate_weights))
        if not isinstance(self.horizon, int) or self.horizon < 1 or self.max_iter < 1:
            raise ValueError('Positive integer horizon and iteration limit required')
        values=[self.terminal_multiplier,*self.rate_weights,self.slack_quadratic,
                self.slack_linear,self.eps_abs,self.eps_rel,self.actuator_tolerance,self.rho]
        if len(self.rate_weights)!=2 or not np.all(np.isfinite(values)) or min(values)<=0:
            raise ValueError('MPC weights/tolerances must be finite and positive')


def steering_chart(x, wheelbase):
    v=max(float(x[2]),.1); omega=float(x[4]); z=wheelbase*omega/v
    gradient=np.zeros(5)
    gradient[4]=wheelbase/v/(1+z*z)
    if x[2]>.1:
        gradient[2]=-wheelbase*omega/v**2/(1+z*z)
    return float(np.arctan(z)),gradient


class MPCController:
    def __init__(self, ref, A, B, config=None, limits=DEFAULT_LIMITS,
                 track_half_width=5., Q=None, R=None):
        try:
            import osqp
        except ImportError as exc:
            raise ImportError('MPC requires OSQP; install requirements.lock.txt') from exc
        self.osqp=osqp; self.ref=ref; self.A=A; self.B=B
        self.config=config or MPCConfig(); self.limits=limits; self.width=track_half_width
        self.Q=Q_LQR if Q is None else Q; self.R=R_LQR if R is None else R
        self.workspaces={}; self.previous_solution=None; self.previous_horizon=None
        key=(self.config,limits,track_half_width,tuple(self.Q.flat),tuple(self.R.flat))
        self.templates=ref.setdefault('_mpc_templates',{}).setdefault(key,{})

    def template(self,k):
        if k in self.templates:
            return self.templates[k]
        c=self.config; L=self.limits; N=min(c.horizon,len(self.A)-k)
        if N<1: raise ValueError('No future reference samples')
        nx=5*(N+1); nu=2*N; total=nx+nu+6*N
        D=sparse.eye(nu,format='csc')-sparse.diags(np.ones(nu-2),-2,shape=(nu,nu),format='csc')
        W=sparse.kron(sparse.eye(N),sparse.diags(c.rate_weights),format='csc')
        P=sparse.block_diag([sparse.kron(sparse.eye(N),self.Q),c.terminal_multiplier*self.Q,
                             sparse.kron(sparse.eye(N),self.R)+D.T@W@D,
                             c.slack_quadratic*sparse.eye(6*N)],format='csc')*2
        rows=[];cols=[];data=[];lo=[];hi=[]
        def add(entries,lower,upper):
            row=len(lo)
            for col,value in entries:
                rows.append(row);cols.append(col);data.append(float(value))
            lo.append(lower);hi.append(upper)
        for j in range(5): add([(j,1.)],0.,0.)
        X=self.ref['states']; U=self.ref['u_ref']
        for i in range(N):
            defect=predict_state_ctrv(X[k+i],DT,U[k+i])-X[k+i+1]
            defect[3]=wrap_to_pi(defect[3])
            for j in range(5):
                entries=[(5*(i+1)+j,1.)]
                entries += [(5*i+t,-self.A[k+i][j,t]) for t in range(5)]
                entries += [(nx+2*i+t,-self.B[k+i][j,t]) for t in range(2)]
                add(entries,defect[j],defect[j])
        for i in range(N):
            add([(nx+2*i,1.)],-L.max_braking-U[k+i,0],L.max_accel-U[k+i,0])
            add([(nx+2*i+1,1.)],-L.max_yaw_accel-U[k+i,1],L.max_yaw_accel-U[k+i,1])
        for i in range(1,N+1):
            x=X[k+i]; prev=X[k+i-1]; delta,gdelta=steering_chart(x,L.wheelbase)
            dprev,gprev=steering_chart(prev,L.wheelbase)
            lateral=np.zeros(5);lateral[2]=x[4];lateral[4]=x[2]
            normal=np.zeros(5);normal[:2]=[-np.sin(x[3]),np.cos(x[3])]
            quantities=[(x[2],np.eye(5)[2],0.,L.max_speed,L.max_speed,None),
                        (x[4],np.eye(5)[4],-L.max_yaw_rate,L.max_yaw_rate,L.max_yaw_rate,None),
                        (x[2]*x[4],lateral,-L.max_lateral_accel,L.max_lateral_accel,L.max_lateral_accel,None),
                        (delta,gdelta,-L.max_steering,L.max_steering,L.max_steering,None),
                        ((delta-dprev)/DT,gdelta/DT,-L.max_steering_rate,L.max_steering_rate,L.max_steering_rate,-gprev/DT),
                        (0.,normal,-self.width,self.width,self.width,None)]
            for group,(base,gradient,lower,upper,scale,previous) in enumerate(quantities):
                entries=[(5*i+t,gradient[t]/scale) for t in range(5)]
                if previous is not None:
                    entries += [(5*(i-1)+t,previous[t]/scale) for t in range(5)]
                slack=nx+nu+6*(i-1)+group
                add(entries+[(slack,-1.)],-np.inf,(upper-base)/scale)
                add([(col,-value) for col,value in entries]+[(slack,-1.)],-np.inf,(base-lower)/scale)
        for i in range(6*N): add([(nx+nu+i,1.)],0.,np.inf)
        C=sparse.csc_matrix((data,(rows,cols)),shape=(len(lo),total))
        q=np.zeros(total);q[nx+nu:]=c.slack_linear
        self.templates[k]=(N,nx,nu,sparse.triu(P,format='csc'),C,np.array(lo),np.array(hi),q,D,W)
        return self.templates[k]

    def command(self,k,state,previous_control,fallback):
        start=perf_counter(); c=self.config
        N,nx,nu,P,C,lower,upper,q,D,W=self.template(k)
        lower=lower.copy();upper=upper.copy();q=q.copy()
        error=np.asarray(state)-self.ref['states'][k];error[3]=wrap_to_pi(error[3])
        lower[:5]=upper[:5]=error
        ff=self.ref['u_ref'][k:k+N].reshape(-1)
        difference=D@ff;difference[:2]-=previous_control
        q[nx:nx+nu]=2*(D.T@(W@difference))
        warm=N in self.workspaces
        if not warm:
            workspace=self.osqp.OSQP()
            workspace.setup(P=P,q=q,A=C,l=lower,u=upper,verbose=False,
                            warm_starting=True,adaptive_rho=False,rho=c.rho,
                            eps_abs=c.eps_abs,eps_rel=c.eps_rel,max_iter=c.max_iter,
                            polishing=True,check_termination=25)
            self.workspaces[N]=workspace
        else:
            workspace=self.workspaces[N]
            workspace.update(q=q,Ax=C.data,l=lower,u=upper)
        # Shift the previous receding-horizon solution; no warm state leaks
        # across simulations because workspaces belong to this controller.
        if self.previous_solution is not None and self.previous_horizon==N:
            old=self.previous_solution
            guess=old.copy()
            guess[:nx]=np.vstack([old[:nx].reshape(N+1,5)[1:],old[:nx].reshape(N+1,5)[-1:]]).reshape(-1)
            guess[nx:nx+nu]=np.vstack([old[nx:nx+nu].reshape(N,2)[1:],old[nx:nx+nu].reshape(N,2)[-1:]]).reshape(-1)
            workspace.warm_start(x=guess,y=np.zeros(C.shape[0]))
        if self.previous_solution is None and warm:
            workspace.warm_start(x=np.zeros(C.shape[1]),y=np.zeros(C.shape[0]))
        try:
            result=workspace.solve(raise_error=False)
        except Exception as exc:
            self.previous_solution=None;self.previous_horizon=None
            bounded=np.clip(fallback,[-self.limits.max_braking,-self.limits.max_yaw_accel],
                            [self.limits.max_accel,self.limits.max_yaw_accel])
            return bounded,{'status':f'solver error: {type(exc).__name__}', 'solved':False,
                            'fallback':True,'iterations':0,'primal_residual':None,
                            'dual_residual':None,'slack_max':None,'slack_mean':None,
                            'horizon':N,'warm_start':warm,'computation_s':perf_counter()-start}
        info=result.info
        finite=result.x is not None and np.all(np.isfinite(result.x))
        command=self.ref['u_ref'][k]+result.x[nx:nx+2] if finite else np.asarray(fallback).copy()
        bounds_lo=np.array([-self.limits.max_braking,-self.limits.max_yaw_accel])
        bounds_hi=np.array([self.limits.max_accel,self.limits.max_yaw_accel])
        valid=(info.status_val==1 and finite and np.all(command>=bounds_lo-c.actuator_tolerance)
               and np.all(command<=bounds_hi+c.actuator_tolerance))
        if valid:
            self.previous_solution=result.x.copy();self.previous_horizon=N
            command=np.clip(command,bounds_lo,bounds_hi)
        else:
            command=np.clip(fallback,bounds_lo,bounds_hi)
            self.previous_solution=None;self.previous_horizon=None
        slack=np.maximum(result.x[nx+nu:],0.) if valid else None
        return command,{'status':info.status,'solved':bool(valid),'fallback':not valid,
                        'iterations':int(info.iter),'primal_residual':float(info.prim_res),
                        'dual_residual':float(info.dual_res),'slack_max':float(slack.max()) if valid else None,
                        'slack_mean':float(slack.mean()) if valid else None,
                        'horizon':N,'warm_start':warm,'computation_s':perf_counter()-start}
