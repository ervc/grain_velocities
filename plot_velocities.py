import numpy as np
import matplotlib.pyplot as plt

import constants as const

def main(velfile: str):
    eqvels = np.load(velfile)
    vx = eqvels['vx']
    vy = eqvels['vy']
    vz = eqvels['vz']
    x = eqvels['x'][:,:,1024]
    y = eqvels['y'][:,:,1024]
    z = eqvels['z'][:,:,1024]

    r = np.sqrt(x*x + y*y)
    phi = np.arctan2(y,x)
    vr = (x*vx + y*vy)/r
    vphi = (x*vy - y*vx)/r/r

    fig,ax = plt.subplots()
    
    from matplotlib.colors import Normalize
    im = ax.pcolormesh(r/const.AU,z,vr,norm=Normalize())
    ct = ax.contour(r/const.AU,z,vr,levels=[0])
    plt.colorbar(im)

    plt.show()



def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('velfile')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.velfile)