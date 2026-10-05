"""Local Gaussian tightening and a covariance-based speed governor.

Margins are model-based risk surrogates, not calibrated safety guarantees.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from ctrv.noise import calculate_Q
from ctrv.constants import DT


@dataclass(frozen=True)
class UncertaintyPolicy:
    tightening: bool = True
    speed_control: bool = True
    confidence: float = .99
    position_scale_m: float = 2.
    heading_scale_rad: float = .15
    minimum_speed_mps: float = 1.
    friction_reserve: float = .8

    def __post_init__(self):
        values=[self.confidence,self.position_scale_m,self.heading_scale_rad,self.minimum_speed_mps,self.friction_reserve]
        if (not isinstance(self.tightening,bool) or not isinstance(self.speed_control,bool) or not np.all(np.isfinite(values))
                or not .5 < self.confidence < 1 or min(values[1:])<=0 or self.friction_reserve>1):
            raise ValueError('Invalid uncertainty policy')

    def speed_cap(self,state,covariance,nominal_speed,curvature,mu=1.):
        covariance=np.asarray(covariance,float)
        if covariance.shape==(8,8):
            from ctrv.vehicle_estimation import validate_vehicle_covariance
            validate_vehicle_covariance(covariance)
            covariance=covariance[:5,:5]
        validate_covariance(covariance)
        if not np.isfinite(nominal_speed) or nominal_speed<0 or not np.isfinite(curvature) or not np.isfinite(mu) or mu<=0:
            raise ValueError("Finite nominal speed, curvature and positive friction required")
        if not self.speed_control:return float(nominal_speed),0.,0.
        normal=np.array([-np.sin(state[3]),np.cos(state[3])])
        sigma_position=np.sqrt(max(0.,normal@covariance[:2,:2]@normal))
        sigma_heading=np.sqrt(max(0.,covariance[3,3]))
        uncertainty=(sigma_position/self.position_scale_m)**2+(sigma_heading/self.heading_scale_rad)**2
        cap=max(self.minimum_speed_mps,nominal_speed/np.sqrt(1+uncertainty))
        friction_speed=np.sqrt(self.friction_reserve*mu*9.81/max(abs(curvature),1e-8))
        return float(min(nominal_speed,cap,friction_speed)),float(sigma_position),float(sigma_heading)


def validate_covariance(P):
    P=np.asarray(P,float)
    if P.shape!=(5,5) or not np.all(np.isfinite(P)) or not np.allclose(P,P.T,atol=1e-8) or np.min(np.linalg.eigvalsh(P))<-1e-8:
        raise ValueError('Finite symmetric PSD 5-state covariance required')


def propagate_covariances(P,A_sequence,states,q_a,q_alpha,dt):
    validate_covariance(P)
    result=[np.asarray(P,float).copy()]
    for A,x in zip(A_sequence,states):
        next_P=A@result[-1]@A.T+calculate_Q(dt,q_a,q_alpha,x[3])
        result.append(.5*(next_P+next_P.T))
    return result


def tighten_bounds(controller,k,upper,covariance,q_a,q_alpha,policy,speed_cap):
    """Tighten each soft envelope side using z*sqrt(g P g.T).

    Steering-rate variance retains adjacent-state cross covariance. Large
    margins remain visible through slack rather than being silently capped.
    """
    N=controller.template(k)[0];X=controller.ref['states'];A=controller.A[k:k+N]
    covs=propagate_covariances(covariance,A,X[k:k+N],q_a,q_alpha,DT)
    start=5+5*N+2*N;z=norm.ppf(policy.confidence) if policy.tightening else 0.;margins=[]
    from ctrv.mpc import steering_chart
    L=controller.limits
    for i in range(1,N+1):
        x=X[k+i];_,gd=steering_chart(x,L.wheelbase);_,gp=steering_chart(X[k+i-1],L.wheelbase)
        glat=np.zeros(5);glat[2]=x[4];glat[4]=x[2]
        normal=np.zeros(5);normal[:2]=[-np.sin(x[3]),np.cos(x[3])]
        gradients=[np.eye(5)[2],np.eye(5)[4],glat,gd,gd/DT,normal]
        scales=[L.max_speed,L.max_yaw_rate,L.max_lateral_accel,L.max_steering,L.max_steering_rate,controller.width]
        for j,(g,scale) in enumerate(zip(gradients,scales)):
            var=g@covs[i]@g
            if j==4:
                old=-gp/DT
                var+=old@covs[i-1]@old+2*g@A[i-1]@covs[i-1]@old
            margin=float(z*np.sqrt(max(0.,var))/scale)
            upper[start+12*(i-1)+2*j:start+12*(i-1)+2*j+2]-=margin
            margins.append(margin)
        # Apply a separate speed ceiling; lower-speed side remains >=0 plus margin.
        if policy.speed_control:
            upper[start+12*(i-1)]+= (min(L.max_speed,speed_cap)-L.max_speed)/L.max_speed
    return float(max(margins)),covs
