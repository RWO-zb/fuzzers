import sys

if len(sys.argv) == 2:
    file_path = sys.argv[1]
else:
    file_path = 'parameters.txt'

f = open(file_path, 'w')

# seeds used across the use-cases (except CARLA)
seeds = [2021, 42, 2023, 20, 0]

# pre-study in RQ3 that shows the little impact of tau on MDPFuzz
k = 10
gamma = 0.01
taus = [0.1, 1.0]
for s in seeds:
    for t in taus:
        print(k, t, gamma, s, file=f)

# configurations studied in RQ3 (parameters, seeds)
tau = 0.01
seeds = [2021, 42, 2023, 20, 0]
ks = [6, 8, 10, 12, 14]
gammas = [0.05, 0.1, 0.15, 0.2]
for s in seeds:
    for k in ks:
        for g in gammas:
            print(k, tau, g, s, file=f)

f.close()
