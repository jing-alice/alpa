
def f1():
  print("*******")
  a = [i for i in range(10)]
  yield from a
 

def f2():
  a = [i for i in range(10)]
  yield a
  

a = f1()

print(next(a))

print(next(a))

print(next(a))

print(next(a))

# def mygen():
#   x = yield 1  
#   print('Received:', x)

# g = mygen()
# next(g) # 启动生成器
# g.send(10)


def mygen():
  x = yield 1
  x = yield x 
  x = yield x
  print('Received:', x)

g = mygen()
print(next(g)) # 1
print(g.send(10)) # 10  
print(g.send(20))
g.send(30)


# contextmanager

from contextlib import contextmanager

@contextmanager
def open_file(filename):
  f = open(filename)
  try:
    yield f
  finally:
    f.close()

with open_file('file') as f:
  # do something with f
  f.write(0)