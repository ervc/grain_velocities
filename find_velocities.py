import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',type=int,default=3)
    parser.add_argument('--mplan',type=int,default=300)
    parser.add_argument('--distro',type=str,default='')
    parser.add_argument('-s','--grainsize',type=float,default=0.1)
    parser.add_argument('-o','--output',type=str,default='')
    parser.add_argument('-v','--verbose',action='store_true')
    
    return parser.parse_args()

def get_fargodir(alpha, mplan, distro) -> str:
    fargodir = f'/Users/ericvc/fargo/outputs/alpha{alpha}_mplan{mplan}'
    if distro != '':
        fargodir+=f'_{distro}'
    return f'{fargodir}/'

def make_directory(outfile: str):
    import os
    os.makedirs(os.path.dirname(outfile),exist_ok=True)
    

def main(alpha: int = 3, mplan: int=300, distro: str='',
         grainsize: float=0.1, output: str='', verbose: bool=False):
    from solver import full_grid_solve, rz_solver, midplane_solver
    from model import Model

    fargodir = get_fargodir(alpha, mplan, distro)
    nout: str = '50'
    if (alpha==3) and (mplan>=200): nout='avg'
    if (alpha==4) and (mplan>=100): nout='avg'
    print('Creating Model...')
    print(f'{alpha = }, {mplan = }, {nout = }')
    print(f'Using fargo directory:\n    {fargodir}')
    model = Model(mplan, fargodir, nout)
    print('Solving...')
    velocities = rz_solver(model,grainsize,verbose=verbose)
    print('Done!')
    if output=='':
        output = f'outputs/alpha{alpha}_mplan{mplan}'
        if distro!='':
            output += f'_{distro}'
        output+=f'/grain{grainsize}_rz.npz'
    make_directory(output)
    print(f'Saving output to:\n    {output}')
    np.savez(output, vx=velocities[0], vy=velocities[1], vz=velocities[2], x=model.xx, y=model.yy, z=model.zz)

if __name__ == "__main__":
    args = parse_args()
    main(alpha=args.alpha, mplan=args.mplan, distro=args.distro, 
         grainsize=args.grainsize, output=args.output, verbose=args.verbose)