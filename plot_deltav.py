import numpy as np
import matplotlib.pyplot as plt

import constants as const

grainsizes = [0.001,0.005,0.01,0.05,0.1,0.5,1.0]

from matplotlib.colors import LogNorm
norm = LogNorm(1e-3,1e5)
cmap='RdYlGn_r'

def read_grainfile(velfile: str):
    eqvels = np.load(velfile)
    s_ = np.s_[80:175, 1024-128:1024+128]
    vx = eqvels['vx']#[s_]
    vy = eqvels['vy']#[s_]
    vz = eqvels['vz']#[s_]
    x = eqvels['x'][:,:,1024]#[-1][s_]
    y = eqvels['y'][:,:,1024]#[-1][s_]
    z = eqvels['z'][:,:,1024]#[-1][s_]

    return x,y,z,vx,vy,vz

def plot_onespot(directory: str):
    ngrain = len(grainsizes)
    fig,axs = plt.subplots(1,4,sharex=True, sharey=True,figsize=(12,4))
    bigdvx = np.ones((ngrain,ngrain))*1.e-16
    bigdvy = np.ones((ngrain,ngrain))*1.e-16
    bigdvz = np.ones((ngrain,ngrain))*1.e-16
    bigdvtot = np.ones((ngrain,ngrain))*1.e-16
    s_ = np.s_[-1,128] # middle of subslice
    for j in range(ngrain):
        velfile1 = directory+f'/grain{grainsizes[j]}_midplane.npz'
        x,y,z, g1vx, g1vy, g1vz = read_grainfile(velfile1)
        for i in range(ngrain):
            velfile2 = directory+f'/grain{grainsizes[i]}_midplane.npz'
            x,y,z, g2vx, g2vy, g2vz = read_grainfile(velfile2)
            dvx = np.abs(g1vx[s_]-g2vx[s_])
            dvy = np.abs(g1vy[s_]-g2vy[s_])
            dvz = np.abs(g1vz[s_]-g2vz[s_])
            dvtot = np.sqrt(dvx**2 + dvy**2 + dvz**2)
            bigdvx[j,i] = max(dvx,1.e-16)
            bigdvy[j,i] = max(dvy,1.e-16)
            bigdvz[j,i] = max(dvz,1.e-16)
            bigdvtot[j,i] = max(dvtot,1.e-16)
    fig.suptitle(f'x = {x[s_]/const.AU:.1f}, y = {y[s_]/const.AU:.1f}, z = {z[s_]/const.AU:.1f} au')
    levels=10
    ax=axs[0]
    ax.pcolormesh(grainsizes,grainsizes,bigdvx,cmap=cmap,norm=norm) #,levels=levels)
    ax.set(
        title='dvx',
        xscale='log',
        yscale='log',
        aspect='equal',
        xlim=(1e-3,1e0),
        ylim=(1e-3,1e0),
    )
    ax=axanes[1]
    ax.pcolormesh(grainsizes,grainsizes,bigdvy,cmap=cmap,norm=norm) #,levels=levels)
    ax.set(
        title='dvy',
        xscale='log',
        yscale='log',
        aspect='equal',
        xlim=(1e-3,1e0),
        ylim=(1e-3,1e0),
    )
    ax=axs[2]
    ax.pcolormesh(grainsizes,grainsizes,bigdvz,cmap=cmap,norm=norm) #,levels=levels)
    ax.set(
        title='dvz',
        xscale='log',
        yscale='log',
        aspect='equal',
        xlim=(1e-3,1e0),
        ylim=(1e-3,1e0),
    )
    ax=axs[3]
    ax.pcolormesh(grainsizes,grainsizes,bigdvtot,cmap=cmap,norm=norm) #,levels=levels)
    ax.set(
        title='dvtot',
        xscale='log',
        yscale='log',
        aspect='equal',
        xlim=(1e-3,1e0),
        ylim=(1e-3,1e0),
    )
    model = directory.split('/')[-1]
    if model=='':
        model = directory.split('/')[-2]
    plt.savefig(f'figs/{model}_onespot.png',bbox_inches='tight')
    plt.show()

def plot_midplane(directory: str):
    ngrain = len(grainsizes)
    fig,axs = plt.subplots(ngrain, ngrain, sharex=True, sharey=True,
                           figsize=(12,12),layout='constrained')
    
    for j in range(ngrain):
        velfile1 = directory+f'/grain{grainsizes[j]}_midplane.npz'
        x,y,z, g1vr, g1vp, g1vz = read_grainfile(velfile1)
        for i in range(j+1):
            velfile2 = directory+f'/grain{grainsizes[i]}_midplane.npz'
            x,y,z, g2vr, g2vp, g2vz = read_grainfile(velfile2)
            ax = axs[j,i]
            dvr = np.abs(g1vr-g2vr)
            dvp = np.abs(g1vp-g2vp)
            dvz = np.abs(g1vz-g2vz)
            dvel = np.sqrt(dvr**2 + dvp**2 + dvz**2)
            ax.pcolormesh(x/const.AU,y/const.AU,dvel,cmap=cmap,norm=norm)
            ax.set_title(f'{grainsizes[j],grainsizes[i]}')
    model = directory.split('/')[-1]
    if model=='':
        model = directory.split('/')[-2]
    plt.savefig(f'figs/{model}_midplane.png',bbox_inches='tight')
    # plt.show()

def plot_rz(directory: str):
    ngrain = len(grainsizes)
    fig,axs = plt.subplots(ngrain, ngrain, sharex=True, sharey=True,
                           figsize=(12,12),layout='constrained')
    
    for j in range(ngrain):
        velfile1 = directory+f'/grain{grainsizes[j]}_rz.npz'
        x,y,z, g1vr, g1vp, g1vz = read_grainfile(velfile1)
        for i in range(j+1):
            velfile2 = directory+f'/grain{grainsizes[i]}_rz.npz'
            x,y,z, g2vr, g2vp, g2vz = read_grainfile(velfile2)
            ax = axs[j,i]
            dvr = np.abs(g1vr-g2vr)
            dvp = np.abs(g1vp-g2vp)
            dvz = np.abs(g1vz-g2vz)
            dvel = np.sqrt(dvr**2 + dvp**2 + dvz**2)
            ax.pcolormesh(x/const.AU,z/const.AU,dvel,cmap=cmap,norm=norm)
            ax.set_title(f'{grainsizes[j],grainsizes[i]}')
    model = directory.split('/')[-1]
    if model=='':
        model = directory.split('/')[-2]
    plt.savefig(f'figs/{model}_rz.png',bbox_inches='tight')
    # plt.show()

def main(directory, which):
    if which=='midplane':
        plot_midplane(directory)
    elif which=='onespot':
        plot_onespot(directory)
    elif which=='rz':
        plot_rz(directory)
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--which',default='midplane')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))