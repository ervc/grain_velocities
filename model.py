import numpy as np
import numpy.typing as npt

import constants as const
from interpolation import interp3d

class Model():
    R0 = 5.2*const.AU
    M_sun = const.MSUN
    G = const.G
    rho_s = 2.0
    TIME = 1/np.sqrt(G*M_sun/R0/R0/R0)
    R_jup = 69911*1e5
    def __init__(self, mplan: float, fargodir: str, nout: str|int):
        self.fargodir = fargodir
        self.nout = str(nout)
        self.mplan = mplan

        self.read_domain()

        self.xx, self.yy, self.zz = self.get_cartgrid()
        self.rhogrid = self.get_rhogrid()
        self.gasvphi, self.gasvr, self.gasvtheta = self.get_spherevelgrid()
        self.gasvx, self.gasvy, self.gasvz = self.get_cartvelgrid()
        self.omegaframe = 1/self.TIME

        self.M_pl = mplan*const.MEARTH
        x_pl = self.R0
        y_pl = 0.0
        z_pl = 0.0
        self.planet_pos = (x_pl, y_pl, z_pl)
        if mplan==0:
            self.sun_pos = (0.0, 0.0, 0.0)
        else:
            self.sun_pos = (-x_pl*self.M_pl/self.M_sun, 0.0, 0.0)
        

        ### TODO: read in model params

    def get_omega(self, x: float, y: float, z: float) -> float:
        r = np.sqrt(x*x + y*y + z*z)
        return np.sqrt(const.G*const.MSUN/r)/r

    def get_scaleheight(self,x: float,y: float,z: float) -> float:
        ### TODO: make this general
        r = np.sqrt(x*x + y*y + z*z)
        return 0.05*r*(r/const.R0)**(1/4)
    
    def get_soundspeed(self,x: float, y: float,z: float) -> float:
        OM = self.get_omega(x,y,z)
        H  = self.get_scaleheight(x,y,z)
        return H*OM
        
    def get_planet_envelope(self,mplan):
        ### TODO: make this general
        """
        double hillRadius = sma * pow(model->planetmass/3/MSUN,1.0/3.0);
        double soundspeed = get_soundspeed(model,sma);
        double bondiRadius = 2*G*model->planetmass/soundspeed/soundspeed;
        // planet enevlope is min(hillRadius/4, bondiRadius)
        model->planetEnvelope = (hillRadius/4. < bondiRadius) ? hillRadius/4 : bondiRadius;
        """
        sma = const.R0
        m_pl = mplan*const.MEARTH
        hill = sma * (m_pl/3/const.MSUN)**(1/3)
        cs = self.get_soundspeed(sma,0,0)
        bondi = 2*const.G*m_pl/cs/cs
        return min(hill/4, bondi)
        


    def read_domain(self):
        self.phi_edges, self.phi_centers = self.read_domfile(
            self.fargodir+'/domain_x.dat', ghostcells=0, scale=1.
            )
        self.r_edges, self.r_centers = self.read_domfile(
            self.fargodir+'/domain_y.dat', ghostcells=3, scale=const.LEN
            )
        self.theta_edges, self.theta_centers = self.read_domfile(
            self.fargodir+'/domain_z.dat', ghostcells=3, scale=1.
            )
        self.nx = len(self.phi_centers)
        self.ny = len(self.r_centers)
        self.nz = len(self.theta_centers)
        self.shape = (self.nz,self.ny,self.nx)

    def read_domfile(self, filename: str, ghostcells: int = 0, scale: float = 1) -> tuple[npt.NDArray, npt.NDArray]:
        xedges = []
        with open(filename,'r') as f:
            for line in f:
                xedges.append(float(line))
        if ghostcells > 0:
            edges = np.array(xedges[ghostcells:-ghostcells])
        else:
            edges = np.array(xedges)
        edges = edges*scale
        centers = (edges[1:] + edges[:-1])/2
        return edges, centers
    
    def get_spheregrid(self) -> list[npt.NDArray]:
        T = self.theta_centers
        R = self.r_centers
        P = self.phi_centers
        return np.meshgrid(T,R,P,indexing='ij')
    
    def get_cartgrid(self) -> list[npt.NDArray]:
        tt,rr,pp = self.get_spheregrid()
        xx = rr*np.cos(pp)*np.sin(tt)
        yy = rr*np.sin(pp)*np.sin(tt)
        zz = rr*np.cos(tt)
        return [xx,yy,zz]
    
    def read_state(self, state: str) -> npt.NDArray:
        filename = self.fargodir+f'/{state}{self.nout}.dat'
        scale = 1.
        if state=='gasdens':
            scale = const.MASS/const.LEN/const.LEN/const.LEN
        elif 'gasv' in state:
            scale = const.LEN/const.TIME
        arr = np.fromfile(filename).reshape(self.shape)
        return arr*scale
    
    def get_rhogrid(self) -> npt.NDArray:
        return self.read_state('gasdens')
    
    def get_rho(self, x: float, y: float, z: float) -> float:
        rhogrid = self.get_rhogrid()
        domain = (self.phi_centers, self.r_centers, self.theta_centers)
        return interp3d(rhogrid, domain, (x,y,z))
    
    def get_spherevelgrid(self) -> list[npt.NDArray]:
        gasvphi = self.read_state('gasvx')
        gasvr = self.read_state('gasvy')
        gasvtheta = self.read_state('gasvz')
        return [gasvphi, gasvr, gasvtheta]
    
    def get_cartvelgrid(self) -> list[npt.NDArray]:
        gasvphi, gasvr, gasvtheta = self.get_spherevelgrid()
        tt,rr,pp = self.get_spheregrid()

        gasvx = (gasvr*np.cos(pp)*np.sin(tt)
                + -gasvphi*np.sin(pp)*np.sin(tt)
                + gasvtheta*np.cos(pp)*np.cos(tt))
        gasvy = (gasvr*np.sin(pp)*np.sin(tt)
                + gasvphi*np.cos(pp)*np.sin(tt)
                + gasvtheta*np.sin(pp)*np.cos(tt))
        gasvz = (gasvr*np.cos(tt) + -gasvtheta*np.sin(tt))

        return [gasvx,gasvy,gasvz]
    
    def get_cartvel(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        gasvx, gasvy, gasvz = self.get_cartvelgrid()
        domain = (self.phi_centers, self.r_centers, self.theta_centers)
        pos = (x,y,z)
        gvx = interp3d(gasvx,domain,pos)
        gvy = interp3d(gasvy,domain,pos)
        gvz = interp3d(gasvz,domain,pos)
        return gvx,gvy,gvz
    
    def get_all_vel0(self, s: float):
        x = self.xx
        y = self.yy
        z = self.zz
        r = np.sqrt(x*x + y*y)
        phi = np.arctan2(y,x)

        rho_g = self.rhogrid
        cs = self.get_soundspeed(x,y,z)
        tstop = (s*self.rho_s)/(cs*rho_g)

        vkep_norot = np.sqrt(self.G*self.M_sun/r)
        Omega = vkep_norot/r
        Stokes = tstop*Omega

        x_sun, y_sun, z_sun = self.sun_pos
        x_pl, y_pl, z_pl = self.planet_pos

        d = np.sqrt( (x-x_sun)**2 + (y-y_sun)**2 + (z-z_sun)**2 )
        GSUN = self.G*self.M_sun/d/d/d
        Gx = GSUN*(x-x_sun)
        Gy = GSUN*(y-y_sun)
        Gz = GSUN*(z-z_sun)
        dp = np.sqrt( (x-x_pl)**2 + (y-y_pl)**2 + (z-z_pl)**2 )
        GPLAN = self.G*self.M_pl/dp/dp/dp
        Gpx = GPLAN*(x-x_pl)
        Gpy = GPLAN*(y-y_pl)
        Gpz = GPLAN*(z-z_pl)

        gasvphi = self.gasvphi
        gasvr = self.gasvr

        eta = 1 - ((gasvphi+r*self.omegaframe)/vkep_norot)**2
        vr0 = (gasvr - eta*Stokes*vkep_norot)/(1+Stokes*Stokes)
        vp0 = gasvphi - 0.5*Stokes*vr0

        armvr = vr0
        armvphi = vp0
        armvx = armvr*np.cos(phi) - armvphi*np.sin(phi)
        armvy = armvr*np.sin(phi) + armvphi*np.cos(phi)
        acent = armvphi*armvphi/r/r

        vx0 = self.gasvx + tstop*(-Gx - Gpx + 2*armvy*self.omegaframe + x*self.omegaframe*self.omegaframe + x*acent)
        vy0 = self.gasvy + tstop*(-Gy - Gpy - 2*armvx*self.omegaframe + y*self.omegaframe*self.omegaframe + y*acent)
        vz0 = self.gasvz + tstop*(-Gz - Gpz)

        return vx0, vy0, vz0


    
    def get_particle_acceleration(self, s: float, vel: tuple[float, float, float], idx: tuple[int,int,int]):
        vx, vy, vz = vel
        i,j,k = idx 
        slice = np.s_[k,j,i]

        x = self.xx[slice]
        y = self.yy[slice]
        z = self.zz[slice]
        r = np.sqrt(x*x + y*y)

        rho_g = self.rhogrid[slice]
        cs = self.get_soundspeed(x,y,z)
        tstop = (s*self.rho_s)/(cs*rho_g)
        invts = 1/tstop

        x_sun, y_sun, z_sun = self.sun_pos
        x_pl, y_pl, z_pl = self.planet_pos
        d = np.sqrt( (x-x_sun)**2 + (y-y_sun)**2 + (z-z_sun)**2 )
        GSUN = self.G*self.M_sun/d/d/d
        Gx = GSUN*(x-x_sun)
        Gy = GSUN*(y-y_sun)
        Gz = GSUN*(z-z_sun)
        dp = np.sqrt( (x-x_pl)**2 + (y-y_pl)**2 + (z-z_pl)**2 )
        GPLAN = self.G*self.M_pl/dp/dp/dp
        Gpx = GPLAN*(x-x_pl)
        Gpy = GPLAN*(y-y_pl)
        Gpz = GPLAN*(z-z_pl)

        vphi = (x*vy - y*vx)/r
        ax = (
            invts*self.gasvx[slice] - invts*vx
            - Gx - Gpx
            + 2*vy*self.omegaframe + x*self.omegaframe*self.omegaframe
            + vphi*vphi/r/r*x
        )
        ay = (
            invts*self.gasvy[slice] - invts*vy
            - Gy - Gpy
            + -2*vx*self.omegaframe + y*self.omegaframe*self.omegaframe
            + vphi*vphi/r/r*y
        )
        az = (
            invts*self.gasvz[slice] - invts*vz
            - Gz - Gpz
        )

        return ax,ay,az
    
    def interp_particle_acceleration(self, s: float, vel: tuple[float, float, float], cartpos: tuple[float, float, float]):
        vx,vy,vz = vel
        x,y,z = cartpos
        r = np.sqrt(x*x + y*y)
        domain = (self.phi_centers, self.r_centers, self.theta_centers)

        rho_g = interp3d(self.rhogrid, domain, cartpos)
        gasvx = interp3d(self.gasvx, domain, cartpos)
        gasvy = interp3d(self.gasvy, domain, cartpos)
        gasvz = interp3d(self.gasvz, domain, cartpos, flip=True)
        cs = self.get_soundspeed(x,y,z)
        tstop = (s*self.rho_s)/(cs*rho_g)
        invts = 1/tstop

        x_sun, y_sun, z_sun = self.sun_pos
        x_pl, y_pl, z_pl = self.planet_pos
        d = np.sqrt( (x-x_sun)**2 + (y-y_sun)**2 + (z-z_sun)**2 )
        GSUN = self.G*self.M_sun/d/d/d
        Gx = GSUN*(x-x_sun)
        Gy = GSUN*(y-y_sun)
        Gz = GSUN*(z-z_sun)
        dp = np.sqrt( (x-x_pl)**2 + (y-y_pl)**2 + (z-z_pl)**2 )
        GPLAN = self.G*self.M_pl/dp/dp/dp
        Gpx = GPLAN*(x-x_pl)
        Gpy = GPLAN*(y-y_pl)
        Gpz = GPLAN*(z-z_pl)

        vphi = (x*vy - y*vx)/r
        ax = (
            invts*gasvx - invts*vx
            - Gx - Gpx
            + 2*vy*self.omegaframe + x*self.omegaframe*self.omegaframe
            + vphi*vphi/r/r*x
        )
        ay = (
            invts*gasvy - invts*vy
            - Gy - Gpy
            + -2*vx*self.omegaframe + y*self.omegaframe*self.omegaframe
            + vphi*vphi/r/r*y
        )
        az = (
            invts*gasvz - invts*vz
            - Gz - Gpz
        )

        return ax,ay,az

    
    def get_particle_jacobian(self, s: float, vel: tuple[float, float, float], idx: tuple[int, int, int]):
        vx, vy, vz = vel
        i,j,k = idx 
        slice = np.s_[k,j,i]

        x = self.xx[slice]
        y = self.yy[slice]
        z = self.zz[slice]
        r = np.sqrt(x*x + y*y)
        r3 = r*r*r

        rho_g = self.rhogrid[slice]
        cs = self.get_soundspeed(x,y,z)
        tstop = (s*self.rho_s)/(cs*rho_g)
        invts = 1/tstop

        vphi = (x*vy - y*vx)/r

        daxdvx = -invts - 2*x*y*vphi/r3
        daxdvy = 2*self.omegaframe + 2*x*x*vphi/r3
        daxdvz = 0.0

        daydvx = -2*self.omegaframe - 2*y*y*vphi/r3
        daydvy = -invts + 2*y*x*vphi/r3
        daydvz = 0.0

        dazdvx = 0.0
        dazdvy = 0.0
        dazdvz = -invts

        return np.array([[daxdvx, daxdvy, daxdvz],
                         [daydvx, daydvy, daydvz],
                         [dazdvx, dazdvy, dazdvz]],
                         dtype=np.double)

    def interp_particle_jacobian(self, s: float, vel: tuple[float, float, float], cartpos: tuple[float, float, float]):
        vx,vy,vz = vel
        x,y,z = cartpos
        r = np.sqrt(x*x + y*y)
        r3 = r*r*r
        domain = (self.phi_centers, self.r_centers, self.theta_centers)

        rho_g = interp3d(self.rhogrid, domain, cartpos)
        cs = self.get_soundspeed(x,y,z)
        tstop = (s*self.rho_s)/(cs*rho_g)
        invts = 1/tstop

        vphi = (x*vy - y*vx)/r

        daxdvx = -invts - 2*x*y*vphi/r3
        daxdvy = 2*self.omegaframe + 2*x*x*vphi/r3
        daxdvz = 0.0

        daydvx = -2*self.omegaframe - 2*y*y*vphi/r3
        daydvy = -invts + 2*y*x*vphi/r3
        daydvz = 0.0

        dazdvx = 0.0
        dazdvy = 0.0
        dazdvz = -invts

        return np.array([[daxdvx, daxdvy, daxdvz],
                         [daydvx, daydvy, daydvz],
                         [dazdvx, dazdvy, dazdvz]],
                         dtype=np.double)