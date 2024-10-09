import cProfile
from find_velocities import main

cProfile.run('main(verbose=True)',sort='tottime')