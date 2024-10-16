import subprocess

grainsizes = [1.0,0.5,0.1,0.05,0.01,0.005,0.001]

for g in grainsizes:
    run = f'python find_velocities.py -v --alpha 3 --mplan 0 -s {g}'
    print(run)
    subprocess.run(run.split())