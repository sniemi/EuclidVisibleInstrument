import multiprocessing

def calc_stuff(value):
    return value**2, value**3, value**4

pool = multiprocessing.Pool(4)
out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10)))

print out1
print
print out2
print
print out3
