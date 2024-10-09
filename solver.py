from model import Model

import numpy as np

from scipy import linalg
tolf = 1.e-16 # tolerance on acceleration
tolx = 1.e-10 # tolerance on velocity
ntrial = 500 # max number of steps

def get_vel0(model: Model, s: float, idx: tuple[int, int, int]):
    i,j,k = idx
    slice = np.s_[k,j,i]
    x = model.xx[slice]
    y = model.yy[slice]
    z = model.zz[slice]
    r = np.sqrt(x*x + y*y)
    phi = np.arctan2(y,x)

    rho_g = model.rhogrid[slice]
    cs = model.get_soundspeed(x,y,z)
    tstop = (s*model.rho_s)/(cs*rho_g)

    vkep_norot = np.sqrt(model.G*model.M_sun/r)
    Omega = vkep_norot/r
    Stokes = tstop*Omega

    x_sun, y_sun, z_sun = model.sun_pos
    x_pl, y_pl, z_pl = model.planet_pos

    d = np.sqrt( (x-x_sun)**2 + (y-y_sun)**2 + (z-z_sun)**2 )
    GSUN = model.G*model.M_sun/d/d/d
    Gx = GSUN*(x-x_sun)
    Gy = GSUN*(y-y_sun)
    Gz = GSUN*(z-z_sun)
    dp = np.sqrt( (x-x_pl)**2 + (y-y_pl)**2 + (z-z_pl)**2 )
    GPLAN = model.G*model.M_pl/dp/dp/dp
    Gpx = GPLAN*(x-x_pl)
    Gpy = GPLAN*(y-y_pl)
    Gpz = GPLAN*(z-z_pl)

    gasvphi = model.gasvphi
    gasvr = model.gasvr

    eta = 1 - ((gasvphi[slice]+r*model.omegaframe)/vkep_norot)**2
    vr0 = (gasvr[slice] - eta*Stokes*vkep_norot)/(1+Stokes*Stokes)
    vp0 = gasvphi[slice] - 0.5*Stokes*vr0

    armvr = vr0
    armvphi = vp0
    armvx = armvr*np.cos(phi) - armvphi*np.sin(phi)
    armvy = armvr*np.sin(phi) + armvphi*np.cos(phi)
    acent = armvphi*armvphi/r/r

    vx0 = model.gasvx[slice] + tstop*(-Gx - Gpx + 2*armvy*model.omegaframe + x*model.omegaframe*model.omegaframe + x*acent)
    vy0 = model.gasvy[slice] + tstop*(-Gy - Gpy - 2*armvx*model.omegaframe + y*model.omegaframe*model.omegaframe + y*acent)
    vz0 = model.gasvz[slice] + tstop*(-Gz - Gpz)

    return vx0, vy0, vz0


def NDNewtSolve(model: Model, s: float, idx: tuple[int, int, int],
                vel0 = None,
                debug: bool=False):
    """Solve for the equilibruim velocity for a given particle within a fargo model

    Args:
        model (Model): Model for fargo output
        s (float): size of the grain in cm
        idx (tuple[int, int, int]): index of particle location (i,j,k)
    """
    i,j,k = idx
    if vel0 is None:
        vx,vy,vz = get_vel0(model, s, idx)
    else:
        vx, vy, vz = vel0
    for n in range(ntrial):
        if debug:
            print(f'n, vx, vy, vz = {n, vx, vy, vz}')
        ax,ay,az = model.get_particle_acceleration(s,(vx,vy,vz), (i,j,k)) # find the acceleration
        if debug:
            print(f'n, ax, ay, az = {n, ax, ay, az}')
        errf = np.abs(ax)+np.abs(ay)+np.abs(az)
        if errf < tolf: return n, (vx,vy,vz)  # if accelerations are less than tolf, end
        jac = model.get_particle_jacobian(s, (vx,vy,vz), (i,j,k))
        A = np.array([ax,ay,az],dtype=np.double)
        dv = linalg.solve(jac,-A)    # solves jac.dv = -A
        errx = np.abs(dv[0]/vx) + np.abs(dv[1]/vy) + np.abs(dv[2]/vz)
        if errx < tolx: return n, (vx,vy,vz)   # if dv is smaller than tolx, end
        vx += dv[0]
        vy += dv[1]
        vz += dv[2]
    print('DID NOT CONVERGE')
    return n, (vx,vy,vz)

def full_grid_solve(model: Model, s: float, verbose: bool=False):
    nx = model.nx
    ny = model.ny
    nz = model.nz
    shape = (3,nz,ny,nx)
    partvels = np.zeros(shape)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if verbose:
                    print(f'{k:5d} {j:5d} {i:5d}',end='\r',flush=True)
                n, vels = NDNewtSolve(model, s, (i,j,k))
                partvels[:,k,j,i] = vels
    if verbose: print('\nDone')

    return partvels

def rz_solver(model: Model, s: float, verbose: bool=False):
    nx = model.nx
    ny = model.ny
    nz = model.nz
    shape = (3,nz,ny)
    partvels = np.zeros(shape)
    vx, vy ,vz = model.get_all_vel0(s)
    for k in range(nz):
        for j in range(ny):
            i = 1024
            if verbose and i%16==0:
                print(f'{k:5d} {j:5d} {i:5d}',end='\r',flush=True)
            n, vels = NDNewtSolve(model, s, (i,j,k), vel0=(vx[k,j,i],vy[k,j,i],vz[k,j,i]))
            partvels[:,k,j] = vels
    if verbose: print('\nDone')

    return partvels

def midplane_solver(model: Model, s: float, verbose: bool=False):
    nx = model.nx
    ny = model.ny
    nz = model.nz
    shape = (3,ny,nx)
    partvels = np.zeros(shape)
    k=nz-1
    vx, vy ,vz = model.get_all_vel0(s)
    for j in range(ny):
        for i in range(nx):
            if verbose and i%16==0:
                print(f'{k:5d} {j:5d} {i:5d}',end='\r',flush=True)
            n, vels = NDNewtSolve(model, s, (i,j,k), vel0=(vx[k,j,i],vy[k,j,i],vz[k,j,i]))
            partvels[:,j,i] = vels
    if verbose: print('\nDone')

    return partvels

def r_solver(model: Model, s: float, verbose: bool=False):
    nx = model.nx
    ny = model.ny
    nz = model.nz
    shape = (3,ny)
    partvels = np.zeros(shape)
    k = nz-1
    for j in range(ny):
        i = 1024
        if verbose:
            print(f'{k:5d} {j:5d} {i:5d}',end='\r',flush=True)
        n, vels = NDNewtSolve(model, s, (i,j,k))
        partvels[:,j] = vels
    if verbose: print('\nDone')

    return partvels

def debug():
    i, j, k = 1024, 150, 0
    model = Model(300,'/Users/ericvc/fargo/outputs/alpha3_mplan300','avg')
    n, (vx, vy, vz) = NDNewtSolve(model, 0.1, (i,j,k), debug=True)
    print(n, vx, vy, vz, sep='\n')

if __name__ == '__main__':
    debug()