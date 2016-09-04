__author__ = 'esthervandenberg'

def read_op(file):
    tmp = open(file).readlines()[1]
    tmp = [o.strip(';').strip('\n') for o in tmp.split(' ') if o != '']
    return tmp[-1]

def compare(files):
    tmps = [float(read_op(file)) for file in files]
    if abs(tmps[0] - tmps[1]) < 0.01:
        print('Converged')

with open('state.txt', 'w') as f:
    f.write('')

if compare(['conlleval_before_one_it', 'conlleval_after_one_it']) == True:
    with open('state.txt', 'w') as f:
        f.write('Converged')