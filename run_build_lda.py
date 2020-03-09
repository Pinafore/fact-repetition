import itertools
import subprocess

header = '''#!/bin/bash
#SBATCH --qos=batch
#SBATCH --time=20:00:00
#SBATCH --job-name=karl_lda
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g

'''

data = ['all']#, 'jeopardy', 'quizbowl']
vocab = [5000, 10000, 20000, 30000, 40000, 50000]
topic = [5, 10, 15]
min_df = [2, 3, 5]

args = itertools.product(data, vocab, topic)

cmd = 'python build_lda.py --data {} --vocab {} --topic {}'
for d, v, t in args:
    with open('build_lda.sh', 'w') as f:
        f.write(header)
        c = cmd.format(d, v, t)
        print(c)
        f.write(cmd.format(d, v, t))
    subprocess.run(['sbatch', 'build_lda.sh'])
