import itertools
import subprocess

header = '''#!/bin/bash
#SBATCH --qos=batch
#SBATCH --time=24:00:00
#SBATCH --job-name=karl_lda
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g

'''

data = ['all']
model = ['gensim']
n_vocab = [50000]
n_topics = [10, 15]
min_df = [2, 5, 10, 20]
max_df = [0.5, 0.8]
args = [data, model, n_vocab, n_topics, min_df, max_df]
cmd = 'python build_lda.py --data {} --model {} --n_vocab {} --n_topics {} --min_df {} --max_df {}'
for d, m, v, t, mi, ma in itertools.product(*args):
    with open('build_lda.sh', 'w') as f:
        f.write(header)
        c = cmd.format(d, m, v, t, mi, ma)
        f.write(c)
        print(c)
    subprocess.run(['sbatch', 'build_lda.sh'])
