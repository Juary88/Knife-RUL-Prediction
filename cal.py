from math import exp, log

X = [i for i in range(79, 110)]

min = 1000000000000
min_x = 0
for x in X:
    tmp = exp(-log(0.5)*(110-x)/5) - exp(log(0.5)*(162-x)/20) - 0.43496
    if tmp < min:
        min_x = x
        min = tmp

print(min_x)
print(min)