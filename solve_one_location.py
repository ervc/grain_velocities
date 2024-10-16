import numpy as np
import matplotlib.pyplot as plt

import constants as const
from find_velocities import get_fargodir
from model import Model
from solver import NDNewtSolve, interp_NewtSolve

grainsizes = np.logspace(-3,3,50)
ngrain = grainsizes.size

def get_nout(alpha, mplan):
    nout = '50'
    if (alpha==3) and (mplan>=200): nout='avg'
    if (alpha==4) and (mplan>=100): nout='avg'
    return nout


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos',default=[1.5707, 5.2, 0.0],nargs=3,type=float)
    parser.add_argument('--coords',default='cyl',choices=['cyl','cart','sphere'])
    parser.add_argument('--alpha',type=int,default=3)
    parser.add_argument('--mplan',type=int,default=300)
    parser.add_argument('--distro',type=str,default='')
    parser.add_argument('-v','--verbose',action='store_true')
    parser.add_argument('-d','--display',action='store_true',help='Show figure created')
    parser.add_argument('--debug',action='store_true')

    return parser.parse_args()

def cyl2sphere(pos):
    phi, r, z = pos
    r = np.sqrt(r*r + z*z)
    theta = np.arccos(z/r)
    return phi, r, theta

def cart2sphere(pos):
    x, y, z = pos
    phi = np.arctan2(y,x)
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z/r)
    return phi, r, theta

def cyl2cart(pos):
    phi, r, z = pos
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x,y,z

def sphere2cart(pos):
    phi, r, theta = pos
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    return x,y,z

def get_spherical_pos(pos, coords):
    if coords=='cyl':
        phi, r, theta = cyl2sphere(pos)
    elif coords=='cart':
        phi, r, theta = cart2sphere(pos)
    else:
        phi, r, theta = pos
    return phi, r, theta

def get_cartesian_pos(pos, coords):
    if coords=='cyl':
        x,y,z = cyl2cart(pos)
    elif coords=='sphere':
        x,y,z = sphere2cart(pos)
    else:
        x,y,z = pos
    return x,y,z

def get_index(pos, coords, model: Model):
    phi, r, theta = get_spherical_pos(pos, coords)
    i = np.argmin(np.abs(phi-model.phi_centers))
    j = np.argmin(np.abs(r-model.r_centers))
    if theta > np.pi/2:
        theta = np.pi-theta
    k = np.argmin(np.abs(theta-model.theta_centers))
    return i,j,k

def plot_results(model: Model, dvs, cartpos, display=True):
    fig,axd = plt.subplot_mosaic(
        """
        MMxy
        MMzt
        """,
        figsize=(9,4), layout='constrained',
    )
    
    ax = axd['M']
    from matplotlib.colors import LogNorm
    im = ax.pcolormesh(model.xx[-1]/const.AU,model.yy[-1]/const.AU,
                       model.rhogrid[-1],
                       norm=LogNorm(),cmap='inferno')
    ax.set(
        xlabel='X',
        ylabel='Y',
        title='midplane density',
        aspect=1,
    )
    plt.colorbar(im,ax=ax,label='gas rho',location='right')
    ax.plot(cartpos[0]/const.AU,cartpos[1]/const.AU,
            c='g',marker='o')

    mindv = -1
    maxdv = 5
    ncont = 2*(maxdv-mindv)+1
    norm = LogNorm(10**mindv,10**maxdv)
    cmap = 'afmhot_r'

    def plot_dv(ax,X,Y,Z,*args,**kwargs):
        ct = ax.contour(X,Y,np.where(Z==0,9**mindv,Z),levels=[1e2,1e3],
                   linestyles=['-','--'],colors='grey')
        im = ax.pcolormesh(X,Y,np.where(Z==0,9**mindv,Z),
                        #    levels=list(np.logspace(mindv,maxdv,ncont)),
                           *args,**kwargs)
        return im, ct

    ax = axd['x']
    im,_ = plot_dv(ax,grainsizes,grainsizes,dvs[:,:,0],norm=norm,cmap=cmap)
    ax.set(
        ylabel='grainsize [cm]',
        title='dvx',
    )

    ax=axd['y']
    plot_dv(ax,grainsizes,grainsizes,dvs[:,:,1],norm=norm,cmap=cmap)
    ax.set(
        title='dvy',
    )

    ax=axd['z']
    plot_dv(ax,grainsizes,grainsizes,dvs[:,:,2],norm=norm,cmap=cmap)
    ax.set(
        xlabel='grainsize [cm]',
        ylabel='grainsize [cm]',
        title='dvz',
    )

    ax=axd['t']
    im,ct = plot_dv(ax,grainsizes,grainsizes,dvs[:,:,3],norm=norm,cmap=cmap)
    ax.set(
        xlabel='grainsize [cm]',
        title='dvtot',
    )
    cbar = plt.colorbar(im,ax=[axd[n] for n in ['x','y','z','t']],label='delta v [cm/s]')
    cbar.add_lines(ct)
    ticks = [1e-2,1e0,1e2]
    St1 = find_Stokes_one(model,cartpos)
    for ax in [axd[n] for n in ['x','y','z','t']]:
        ax.set(
            xscale='log',
            yscale='log',
            aspect=1,
        )
        ax.axhline(St1, c='k', ls='--')
        ax.axvline(St1, c='k', ls='--')

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        

    fig.suptitle(
        f'M_plan = {model.mplan}\n'
        + f'x = {cartpos[0]/const.AU:.1f}, '
        + f'y = {cartpos[1]/const.AU:.1f}, '
        + f'z = {cartpos[2]/const.AU:.1f}'
    )
    figname = (
        f'figs/mplan{model.mplan}_'
        + f'xyz{cartpos[0]/const.AU:.1f}'
        + f'{cartpos[1]/const.AU:.1f}'
        + f'{cartpos[2]/const.AU:.1f}_dv.png'
    )
    print(f"saving to {figname}")
    plt.savefig(figname,bbox_inches='tight')
    if display:
        plt.show()

def get_Stokes(model: Model, s, cartpos):
    from interpolation import interp3d
    rho_g = interp3d(
        model.rhogrid, 
        (model.phi_centers, model.r_centers, model.theta_centers),
        cartpos
    )
    cs = model.get_soundspeed(*cartpos)
    omega = model.get_omega(*cartpos)
    Stokes = (s*model.rho_s)/(cs*rho_g) * omega
    return Stokes

def find_Stokes_one(model: Model, cartpos):
    from interpolation import interp3d
    rho_g = interp3d(
        model.rhogrid, 
        (model.phi_centers, model.r_centers, model.theta_centers),
        cartpos
    )
    cs = model.get_soundspeed(*cartpos)
    omega = model.get_omega(*cartpos)
    # 1 = s * rho_s * omega / cs * rho_g
    # cs * rho_g / rho_s * omega = s
    return (cs*rho_g)/(model.rho_s*omega)

def get_dvs(vels):
    dvs = np.zeros((ngrain,ngrain,4))
    for j in range(ngrain):
        vx1, vy1, vz1 = vels[j]
        for i in range(ngrain):
            vx2, vy2, vz2 = vels[i]
            dvx = np.abs(vx1-vx2)
            dvy = np.abs(vy1-vy2)
            dvz = np.abs(vz1-vz2)
            dvtot = np.sqrt(dvx**2 + dvy**2 + dvz**2)
            dvs[j,i,0] = dvx
            dvs[j,i,1] = dvy
            dvs[j,i,2] = dvz
            dvs[j,i,3] = dvtot
    return dvs

def rescale(pos, coords):
    if coords=='cart':
        x,y,z = pos
        x*=const.AU
        y*=const.AU
        z*=const.AU
        pos=(x,y,z)
    elif coords=='cyl':
        phi,r,z = pos
        r*=const.AU
        z*=const.AU
        pos = (phi,r,z)
    elif coords=='sphere':
        phi,r,theta = pos
        r*=const.AU
        pos = (phi,r,theta)
    else:
        raise ValueError('pos must be in ["cart", "cyl", "sphere"]')
    return pos

def test(model: Model, pos, coords):
    from solver import interp_NewtSolve
    ss = [0.001]
    cartpos = get_cartesian_pos(pos,coords)
    for s in ss:
        interp_NewtSolve(model, s, cartpos, debug=True)
    omega = model.get_omega(*cartpos)
    x,y,z = cartpos
    r = np.sqrt(x*x + y*y)
    vkep_norot = omega/r
    vkep = vkep_norot - r*omega
    phi = np.arctan2(y,x)
    vx = -vkep*np.sin(phi)
    vy = vkep*np.cos(phi)
    print(f'keplerian: {vx = }, {vy = }')
    return



def main(pos, coords, alpha, mplan, distro, verbose, display, debug):
    def vprint(*args,**kwargs):
        if verbose:
            print(*args,**kwargs,flush=True)
    vprint(f'{pos = }')
    pos = rescale(pos,coords)
    fargodir = get_fargodir(alpha, mplan, distro)
    vprint('Making Model')
    model = Model(mplan, fargodir, get_nout(alpha,mplan))
    # return test(model, pos, coords)
    cartpos = get_cartesian_pos(pos, coords)
    vels = np.zeros((ngrain,3))
    
    vprint('Solving...')
    for n,s in enumerate(grainsizes):
        _, vels[n] = interp_NewtSolve(model, s, cartpos, debug=debug)
    dvs = get_dvs(vels)

    vprint('plotting...')
    plot_results(model,dvs,get_cartesian_pos(pos,coords),display)
    



if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))