"""Actuator-aware local MPC for the eight-state synthetic vehicle.

Each QP uses the same discrete nonlinear map as the plant and estimator.
The eight normalized soft envelopes remain local approximations. Covariance
margins use local Gaussian surrogates or bounded acceleration quantiles;
there is no claim of joint chance feasibility or nonlinear safety.
"""
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.optimize import root
from ctrv.constants import DT, Q_LQR
from ctrv.model import wrap_to_pi
from ctrv.limits import DEFAULT_LIMITS
from ctrv.mpc import MPCConfig
from ctrv.vehicle import VehicleConfig, derivative
from ctrv.vehicle_estimation import (VehicleNoise, vehicle_linearization,
    vehicle_process_covariance, validate_vehicle_covariance)

Q_VEHICLE=sparse.block_diag([Q_LQR,np.diag([2.,.1,1.])]).toarray()
R_VEHICLE=np.diag([1.,50.])


def equilibrium_state(path_state, curvature, mu, config=VehicleConfig(), limits=DEFAULT_LIMITS):
    """Steady cornering feedforward; body heading accounts for sideslip.

    The horizon's affine dynamics defect accounts for remaining reference
    inconsistency, including transients. A failed equilibrium solve is explicit.
    """
    speed=path_state[2]; rate=speed*curvature
    initial=np.array([config.rear_distance_m*rate,
                      np.arctan(limits.wheelbase*curvature),config.drag_coefficient*speed**2/config.mass_kg])
    def state_for(z):
        vy,delta,a=z
        vx=np.sqrt(max(.25,speed**2-vy**2))
        return np.array([path_state[0],path_state[1],vx,
                         wrap_to_pi(path_state[3]-np.arctan2(vy,vx)),rate,vy,a,delta])
    def residual(z):
        x=state_for(z);f=derivative(x,np.array([z[2],z[1]]),mu,config,limits)
        return f[[2,4,5]]
    solution=root(residual,initial)
    if not solution.success or np.linalg.norm(residual(solution.x))>1e-6:
        raise ValueError('Reference has no converged local steady vehicle equilibrium')
    x=state_for(solution.x)
    if abs(x[7])>limits.max_steering or not -limits.max_braking<=x[6]<=limits.max_accel:
        raise ValueError('Steady vehicle reference exceeds actuator envelope')
    return x


def build_vehicle_path(reference, mu=.9, config=VehicleConfig(), limits=DEFAULT_LIMITS):
    states=np.array([equilibrium_state(x,k,mu,config,limits)
                     for x,k in zip(reference['states'],reference['curvature'])])
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference['states'][:,:2],axis=0),axis=1))]
    return {'states':states,'arc':arc,'curvature':reference['curvature'].copy(),
            'heading':np.unwrap(states[:,3]),'path_heading':np.unwrap(reference['states'][:,3]),'length':float(arc[-1])}


def rolling_reference(path, progress, cap, horizon=20, limits=DEFAULT_LIMITS,
                      config=VehicleConfig(),mu=.9):
    """Time-parameterize the path locally at the governed speed.

    Steady slip angles approximate inverse saturated tire forces. Acceleration and
    actuator feedforward include the path speed gradient and response lag.
    Remaining transient/reference errors enter the MPC affine defect. A
    reference beyond available friction remains a soft tracking target.
    """
    X=[];s=float(progress)
    nominal_derivative=np.gradient(path['states'][:,2],path['arc'])
    length=config.front_distance_m+config.rear_distance_m
    weights=np.array([config.rear_distance_m,config.front_distance_m])/length
    normal=config.mass_kg*9.81*weights
    stiffness=np.array([config.cornering_front_nprad,config.cornering_rear_nprad])
    for _ in range(horizon+1):
        x=np.array([np.interp(s,path['arc'],path['states'][:,j]) for j in range(8)])
        tangent=np.interp(s,path['arc'],path['path_heading'])
        nominal=max(x[2],.1);v=min(nominal,cap)
        curvature=np.interp(s,path['arc'],path['curvature']);r=v*curvature
        desired_lateral=config.mass_kg*v*r*weights
        reserve=mu*normal
        slip=reserve/stiffness*np.arctanh(np.clip(desired_lateral/reserve,-.98,.98))
        vy=config.rear_distance_m*r-v*np.tan(slip[1])
        delta=np.arctan2(vy+config.front_distance_m*r,max(v,.5))+slip[0]
        dvds=np.interp(s,path['arc'],nominal_derivative) if nominal<cap else 0.
        desired_acceleration=v*dvds
        # Account for drag and front-force rotation when supplying longitudinal
        # force. This is feedforward only; tire integration still limits force.
        acceleration=(desired_acceleration+config.drag_coefficient*v*v/config.mass_kg-r*vy
                      +desired_lateral[0]*np.sin(delta)/config.mass_kg)/(weights[0]*np.cos(delta)+weights[1])
        x[2]=v;x[3]=wrap_to_pi(tangent-np.arctan2(vy,max(v,.5)));x[4]=r;x[5]=vy
        x[6]=np.clip(acceleration,-limits.max_braking,limits.max_accel)
        x[7]=np.clip(delta,-limits.max_steering,limits.max_steering)
        X.append(x);s=min(path['length'],s+DT*np.hypot(v,vy))
    X=np.array(X)
    U=X[:-1][:,[6,7]].copy()
    U[:,0]+=config.acceleration_tau_s*np.diff(X[:,6])/DT
    U[:,1]+=config.steering_tau_s*np.diff(X[:,7])/DT
    U=np.clip(U,[-limits.max_braking,-limits.max_steering],[limits.max_accel,limits.max_steering])
    return X,U


def envelope(state,limits=DEFAULT_LIMITS,mu=.9,config=VehicleConfig()):
    rate=derivative(state,state[[6,7]],mu,config,limits)
    return np.array([np.hypot(state[2],state[5]),state[4],rate[5]+state[2]*state[4],state[7],rate[2]-state[4]*state[5],rate[4]])


def envelope_gradients(state,limits=DEFAULT_LIMITS,mu=.9,config=VehicleConfig()):
    base=envelope(state,limits,mu,config);J=np.zeros((6,8))
    speed=np.hypot(state[2],state[5])
    if speed>1e-8:J[0,[2,5]]=state[[2,5]]/speed
    J[1,4]=1;J[3,7]=1
    # Differentiate the physical lateral acceleration, not v*yaw proxy.
    for j in [2,4,5,6,7]:
        eps=1e-5*max(1.,abs(state[j]));a=state.copy();b=state.copy()
        a[j]+=eps;b[j]-=eps
        J[[2,4,5],j]=(envelope(a,limits,mu,config)[[2,4,5]]-envelope(b,limits,mu,config)[[2,4,5]])/(2*eps)
    return base,J


class VehicleMPC:
    def __init__(self, config=None, vehicle_config=VehicleConfig(),limits=DEFAULT_LIMITS,
                 noise=VehicleNoise(),width=5.,prediction_model="linearized"):
        try:import osqp
        except ImportError as exc:raise ImportError('Vehicle MPC requires OSQP') from exc
        self.osqp=osqp;self.config=config or MPCConfig(rate_weights=(.5,50.));self.vehicle_config=vehicle_config
        if not np.isfinite(width) or width<=0:raise ValueError("Positive finite corridor width required")
        if prediction_model not in ('linearized','bounded'):raise ValueError('Unknown prediction model')
        self.prediction_model=prediction_model
        self.limits=limits;self.noise=noise;self.width=width;self.workspaces={};self.solution=None

    def command(self,state,covariance,X,U,previous_command,mu=.9,policy=None,speed_cap=None):
        start=perf_counter();P0=validate_vehicle_covariance(covariance);c=self.config;L=self.limits
        X=np.asarray(X,float);U=np.asarray(U,float);state=np.asarray(state,float)
        previous_command=np.asarray(previous_command,float)
        N=len(U)
        if (not 1<=N<=c.horizon or X.shape!=(N+1,8) or U.shape!=(N,2) or state.shape!=(8,)
                or previous_command.shape!=(2,) or not all(np.all(np.isfinite(v)) for v in [X,U,state,previous_command])
                or not np.isfinite(mu) or mu<=0):raise ValueError('Invalid finite vehicle MPC horizon or inputs')
        nx=8*(N+1);nu=2*N;n=nx+nu+8*N
        mapped,A,B=vehicle_linearization(X[:-1],U,DT,mu,self.vehicle_config,L)
        # Same physical coordinates, weights and local model for fallback.
        S=c.terminal_multiplier*Q_VEHICLE;gains=[]
        for i in reversed(range(N)):
            gain=np.linalg.solve(R_VEHICLE+B[i].T@S@B[i],B[i].T@S@A[i])
            S=Q_VEHICLE+A[i].T@S@(A[i]-B[i]@gain);S=(S+S.T)/2
            gains.append(gain)
        error=np.asarray(state)-X[0];error[3]=wrap_to_pi(error[3])
        fallback=U[0]-gains[-1]@error
        lo_cmd=np.array([-L.max_braking,-L.max_steering]);hi_cmd=np.array([L.max_accel,L.max_steering])
        fallback=np.clip(fallback,lo_cmd,hi_cmd)
        D=sparse.eye(nu,format='csc')-sparse.diags(np.ones(nu-2),-2,shape=(nu,nu),format='csc')
        W=sparse.kron(sparse.eye(N),sparse.diags(c.rate_weights),format='csc')
        Hessian=2*sparse.block_diag([sparse.kron(sparse.eye(N),Q_VEHICLE),c.terminal_multiplier*Q_VEHICLE,
                  sparse.kron(sparse.eye(N),R_VEHICLE)+D.T@W@D,c.slack_quadratic*sparse.eye(8*N)],format='csc')
        q=np.zeros(n);q[nx+nu:]=c.slack_linear
        difference=D@U.reshape(-1);difference[:2]-=previous_command
        q[nx:nx+nu]=2*D.T@(W@difference)
        rows=[];cols=[];data=[];lower=[];upper=[]
        def add(entries,lo,hi):
            row=len(lower)
            for col,val in entries:rows.append(row);cols.append(col);data.append(float(val))
            lower.append(lo);upper.append(hi)
        for j in range(8):add([(j,1.)],error[j],error[j])
        for i in range(N):
            defect=mapped[i]-X[i+1];defect[3]=wrap_to_pi(defect[3])
            for j in range(8):
                add([(8*(i+1)+j,1.)]+[(8*i+t,-A[i][j,t]) for t in range(8)]
                    +[(nx+2*i+t,-B[i][j,t]) for t in range(2)],defect[j],defect[j])
        for i in range(N):
            for j in range(2):add([(nx+2*i+j,1.)],lo_cmd[j]-U[i,j],hi_cmd[j]-U[i,j])
        covs=[P0];margins=[];quantity_margins=np.zeros(8)
        shifts=np.zeros((N,8));process_cov=np.array([vehicle_process_covariance(b,DT,self.noise) for b in B])
        if self.prediction_model=='bounded':
            from ctrv.bounded_uncertainty import command_moments, acceleration_quantiles
            shifts,process_cov=command_moments(X[:-1],U,DT,mu,self.vehicle_config,L,self.noise)
        mean_error=np.zeros(8);forecast_means=[X[0]]
        for i in range(N):
            covs.append(A[i]@covs[-1]@A[i].T+process_cov[i])
            mean_error=A[i]@mean_error+shifts[i]
            forecast_means.append(X[i+1]+mean_error)
        from scipy.stats import norm
        z=norm.ppf(policy.confidence) if policy is not None and policy.tightening else 0.
        acceleration_bounds=[];empty_acceleration_intervals=0
        for i in range(1,N+1):
            base,G=envelope_gradients(X[i],L,mu,self.vehicle_config)
            normal=np.zeros(8);heading=X[i,3]+np.arctan2(X[i,5],X[i,2])
            normal[:2]=[-np.sin(heading),np.cos(heading)]
            rate_grad=np.eye(8)[7]/DT;old_grad=-rate_grad
            quantities=[(base[0],G[0],0.,min(L.max_speed,speed_cap) if speed_cap is not None else L.max_speed,L.max_speed,None),
                        (base[1],G[1],-L.max_yaw_rate,L.max_yaw_rate,L.max_yaw_rate,None),
                        (base[2],G[2],-L.max_lateral_accel,L.max_lateral_accel,L.max_lateral_accel,None),
                        (base[3],G[3],-L.max_steering,L.max_steering,L.max_steering,None),
                        (base[4],G[4],-L.max_braking,L.max_accel,L.max_braking,None),
                        (base[5],G[5],-L.max_yaw_accel,L.max_yaw_accel,L.max_yaw_accel,None),
                        ((X[i,7]-X[i-1,7])/DT,rate_grad,-L.max_steering_rate,L.max_steering_rate,L.max_steering_rate,old_grad),
                        (0.,normal,-self.width,self.width,self.width,None)]
            for j,(value,g,lo,hi,scale,old) in enumerate(quantities):
                variance=g@covs[i]@g
                if old is not None:variance+=old@covs[i-1]@old+2*g@A[i-1]@covs[i-1]@old
                margin=float(z*np.sqrt(max(0.,variance))/scale)
                # Steering angle/rate are enforced by the physical actuator map.
                # Their uncertainty cannot cause a violation of these limits.
                if j in (3,6):margin=0.
                lower_margin=0. if j==0 else margin  # groundspeed is nonnegative by definition
                if not (j==4 and self.prediction_model=='bounded' and z>0):
                    margins.append(margin);quantity_margins[j]=max(quantity_margins[j],margin)
                entries=[(8*i+t,g[t]/scale) for t in range(8)]
                if old is not None:entries.extend((8*(i-1)+t,old[t]/scale) for t in range(8))
                upper_backoff=margin;lower_backoff=lower_margin
                if j==4 and z>0:
                    if self.prediction_model=='bounded':
                        low_quantile,high_quantile=acceleration_quantiles(forecast_means[i],covs[i],policy.confidence,mu,self.vehicle_config,L)
                        upper_backoff=(high_quantile-value)/scale
                        lower_backoff=(value-low_quantile)/scale
                    else:
                        low_quantile=value-margin*scale;high_quantile=value+margin*scale
                    acceleration_bounds.append([float(low_quantile),float(high_quantile)])
                    quantity_margins[j]=max(quantity_margins[j],upper_backoff,lower_backoff)
                    if self.prediction_model=='bounded':margins.append(max(0.,upper_backoff,lower_backoff))
                    empty_acceleration_intervals+=int(hi-lo+1e-8 < high_quantile-low_quantile)
                slack=nx+nu+8*(i-1)+j
                add(entries+[(slack,-1.)],-np.inf,(hi-value)/scale-upper_backoff)
                add([(col,-val) for col,val in entries]+[(slack,-1.)],-np.inf,(value-lo)/scale-lower_backoff)
        for i in range(8*N):add([(nx+nu+i,1.)],0.,np.inf)
        C=sparse.csc_matrix((data,(rows,cols)),shape=(len(lower),n))
        objective_scale=1./c.slack_quadratic
        H=sparse.triu(Hessian*objective_scale,format='csc');q*=objective_scale
        # Optimize in dimensionless state and command coordinates. Keep every
        # stored zero so the sparse update pattern is identical at each step.
        state_scale=np.array([self.width,self.width,L.max_speed,1.,L.max_yaw_rate,2.,L.max_braking,L.max_steering])
        input_scale=np.array([L.max_braking,L.max_steering])
        variable_scale=np.r_[np.tile(state_scale,N+1),np.tile(input_scale,N),np.ones(8*N)]
        row_scale=np.ones(len(lower))
        row_scale[:8*(N+1)]=1./np.tile(state_scale,N+1)
        row_scale[8*(N+1):8*(N+1)+2*N]=1./np.tile(input_scale,N)
        C.data*=np.repeat(variable_scale,np.diff(C.indptr))*row_scale[C.indices]
        H.data*=np.repeat(variable_scale,np.diff(H.indptr))*variable_scale[H.indices]
        q*=variable_scale;lower=np.asarray(lower)*row_scale;upper=np.asarray(upper)*row_scale
        workspace=self.workspaces.get(N);warm=workspace is not None
        if workspace is None:
            workspace=self.osqp.OSQP()
            # A fixed iteration interval makes rho adaptation independent of
            # machine timing; no wall-clock termination is used.
            workspace.setup(P=H,q=q,A=C,l=lower,u=upper,verbose=False,eps_abs=c.eps_abs,
                            eps_rel=c.eps_rel,max_iter=c.max_iter,rho=c.rho,adaptive_rho=True,
                            adaptive_rho_interval=50,warm_starting=True,polishing=True)
            self.workspaces[N]=workspace
        else:workspace.update(q=q,Ax=C.data,l=lower,u=upper)
        if self.solution is not None and len(self.solution)==n:
            guess=self.solution.copy()
            guess[:nx]=np.vstack([guess[:nx].reshape(N+1,8)[1:],guess[:nx].reshape(N+1,8)[-1:]]).ravel()
            guess[nx:nx+nu]=np.vstack([guess[nx:nx+nu].reshape(N,2)[1:],guess[nx:nx+nu].reshape(N,2)[-1:]]).ravel()
            workspace.warm_start(x=guess,y=np.zeros(C.shape[0]))
        else:workspace.warm_start(x=np.zeros(n),y=np.zeros(C.shape[0]))
        try:
            result=workspace.solve(raise_error=False);info=result.info
            finite=result.x is not None and np.all(np.isfinite(result.x))
            command=U[0]+input_scale*result.x[nx:nx+2] if finite else fallback
            solved=bool(info.status_val==1 and finite and np.all(command>=lo_cmd-c.actuator_tolerance)
                        and np.all(command<=hi_cmd+c.actuator_tolerance))
            diagnostic={'status':info.status,'iterations':int(info.iter),'primal_residual':float(info.prim_res),
                        'dual_residual':float(info.dual_res)}
        except Exception as exc:
            solved=False;diagnostic={'status':f'solver error: {type(exc).__name__}','iterations':0,
                                   'primal_residual':None,'dual_residual':None}
        self.solution=result.x.copy() if solved else None
        command=np.clip(command if solved else fallback,lo_cmd,hi_cmd)
        slack=np.maximum(self.solution[nx+nu:],0.) if solved else None
        diagnostic.update(solved=solved,fallback=not solved,horizon=N,warm_start=warm,
                          slack_max=float(slack.max()) if solved else None,
                          slack_mean=float(slack.mean()) if solved else None,
                          prediction_model=self.prediction_model,acceleration_quantiles=acceleration_bounds,unattainable_acceleration_steps=empty_acceleration_intervals,tightening_max=max(margins),tightening_by_quantity=quantity_margins.tolist(),objective_scale=objective_scale,computation_s=perf_counter()-start)
        return command,diagnostic
