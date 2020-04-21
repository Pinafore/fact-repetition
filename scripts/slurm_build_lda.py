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
n_topics = [40]
min_df = [5]
max_df = [0.5]
args = [data, model, n_vocab, n_topics, min_df, max_df]
cmd = '/fs/clip-quiz/shifeng/anaconda3/bin/python karl/lda.py --data {} --model {} --n_vocab {} --n_topics {} --min_df {} --max_df {}'
for d, m, v, t, mi, ma in itertools.product(*args):
    with open('scripts/build_lda.sh', 'w') as f:
        f.write(header)
        c = cmd.format(d, m, v, t, mi, ma)
        f.write(c)
        print(c)
    subprocess.run(['sbatch', 'scripts/build_lda.sh'])
