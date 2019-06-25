import hashlib
# 111316Ar0000000000000000000000000000000000000000000000000000000000000000
# 111316Arp0000000000000000000000000000000000000000000000000000000000000000
# 1131845Arpit0000000000000000000000000000000000000000000000000000000000000000
# 131845Arpit0000000000000000000000000000000000000000000000000000000000000000'
# 118166969Arpit0000000000000000000000000000000000000000000000000000000000000000
'''https://anders.com/blockchain/blockchain.html'''

found = False
input = 'Arpit'
previous = '0000000000000000000000000000000000000000000000000000000000000000'
x = 0
block = '1'

while not found:
    input_sha = block + str(x) + str(input) + previous
    result = hashlib.sha256(input_sha.encode()).hexdigest()
    if result[:6] == '000000':
        found = True
        print(result)
        print(x)
    else:
        x +=1

found = False
while not found:
    input_sha = block + str(x) + str(input) + previous
    result = hashlib.sha256(input_sha.encode()).hexdigest()
    if result[:7] == '0000000':
        found = True
        print(result)
        print(x)
    else:
        x +=1

found = False
while not found:
    input_sha = block + str(x) + str(input) + previous
    result = hashlib.sha256(input_sha.encode()).hexdigest()
    if result[:8] == '00000000':
        found = True
        print(result)
        print(x)
    else:
        x +=1