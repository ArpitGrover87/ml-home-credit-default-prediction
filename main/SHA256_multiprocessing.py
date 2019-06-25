import hashlib
from joblib import Parallel, delayed


# input = 'Arpit'
# previous = '0000000000000000000000000000000000000000000000000000000000000000'
# #x = 0
# block = '1'
# found = False

def calc_nonce():
    input = 'Arpit'
    previous = '0000000000000000000000000000000000000000000000000000000000000000'
    # x = 0
    block = '1'
    found = False
    x = 5896079044
    while not found and x < 6500000000:
        input_sha = block + str(x) + str(input) + previous
        result = hashlib.sha256(input_sha.encode()).hexdigest()
        if result[:9] == '000000000':
            found = True
            print(x)
            print(result)
        else:
            x += 1


Parallel(n_jobs=-1)(delayed(calc_nonce()))

## morning choice fiction change argue use donate future coyote bonus blue novel

# 1.design
# 2.robot
# 3.unlock
# 4.because
# 5.goat
# 6.cute
# 7.neglect
# 8.awkward
# 9.obey
# 10.hill
# 11.huge
# 12.asset