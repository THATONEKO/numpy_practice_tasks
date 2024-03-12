# 1
import numpy as np


'''print(np.__version__)

# 2
list1 = [12.23, 13.32, 100, 36.32]
arr = np.array(list1)
print(arr)

# 3
arr = np.arange(2, 11).reshape(3, 3)

print(arr)

# 4
x = np.zeros(10)
x[6] = 11
print(x)

# 5
arr = np.arange(12, 39)
arr1 = np.array(arr)

print(arr1)

# 6
arr = np.arange(12, 38)

x = arr[::-1]
print(x)

# 7
arr = np.arange(5, 20, dtype='f')
print(arr)

# 8 and 9
x = np.zeros((5, 5))
x[1:-1, 1:-1] = 1

print(x)

# 10 checkers board
arr = np.ones((3, 3))

arr = np.zeros((8, 8), dtype=int)
arr[1::2, ::2] = 1
arr[::2, 1::2] = 1
print(arr)

# 11
list1 = [i for i in range(10)]
print(list1)

print(np.asarray(list1))

# 12
arr = np.array([10, 30])
print(arr)

x = np.append(arr, [20])
print(x)

# 13
x = np.empty((3, 4))
print(x)

e = np.full((3, 3), 6)

print(e)

# 14 Fahrenheit degrees to Centigrade degrees
fvalues = [0, 12, 45.21, 34, 99.91, 32]

F = np.array(fvalues)
print(F)

print(np.round((5 * F / 9 - 5 * 32 / 9), 2))

# 15
x = np.sqrt([1 + 0j])

y = np.sqrt([0 + 1j])

print(x.real)
print(y.real)

print(x.imag)
print(y.imag)

# 16

arr = np.arange(1, 10)
x = np.array(arr)

print(x.size)
print(x.itemsize)
print(x.nbytes)

# 17 different values
arr = np.array([])

arr1 = np.array([10, 20, 30, 40, 50, 60])
print(arr1)

arr2 = np.array([30, 60])
print(arr2)

print(np.in1d(arr1, arr2))

# 18 common values
arr_test = ([])

arr = np.array([10, 20, 30, 40, 50])
arr1 = np.array([10, 30, 50])

x = np.intersect1d(arr, arr1)

print(x)

# 19

arr = np.array([1, 2, 2, 3, 3, 2, 2, 1, 4, 4, 5, 5])
x = np.unique(arr)

# 20

arr1 = ([10, 20, 30, 50, 60, 70])
arr2 = ([20, 40, 60, 80])

print(np.setdiff1d(arr1, arr2))

# 21

arr = np.array([10, 20, 30, 40, 60, 70])
arr2 = np.array([20, 30, 40, 50, 60, 80])

print(np.setxor1d(arr, arr2))

# 22

arr1 = np.array([10, 10, 20, 30, 40])
arr2 = np.array([10, 20, 50, 50, 30])

print(np.union1d(arr1, arr2)

# 23

print(np.all([0, 0, 1, 2, 3, 0, 9]))

# 24

print(np.any([10, -20, 30, 0]))

# 25

a = np.array([1, 2, 3, 4, 5])
print(a)

x = np.tile(a, 6)
print(x)

# 26

a = np.repeat(3, 4)
print(a)

x = np.array([[1, 2], [3, 4]])
print(np.repeat(x, 7))

# 27

arr = np.array([8, 4, 5, 9, 27, 0, 12])

print(np.argmax(arr))
print(np.argmin(arr))

# 28

a = np.array([2, 27])
b = np.array([4, 13])

print(np.greater(a, b))
print(np.greater_equal(a, b))
print(np.less(a, b))
print(np.less_equal(a, b))

# 29

arr = np.array([[7, 3], [2, 1]])

x = np.sort(arr, axis=0)
print(x)

y = np.sort(x, axis=1)
print(y)

# 30

first_names = ('Betsey', 'Shelley', 'Lanell', 'Genesis', 'Margery')
last_names = ('Battle', 'Brien', 'Plotner', 'Stahl', 'Woolum')

x = np.lexsort((first_names, last_names))

print(x)

# 31

x = np.array([[0, 10, 20], [20, 30, 40]])

print(x)

print(x[x > 10])
print(np.nonzero(x > 10))

# 32 saving array to a file

a = np.arange(1, 10)

np.savetxt('file.out', a, delimiter=',')

# 33 memory size

a = np.ones((4, 4))

print(a.size * a.itemsize)

# 34 data type

arr = np.zeros((3, 3), dtype='i')
print(arr)

# 35 dimension

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr.reshape(3, 3))

# 36 flattening an array

arr = np.array([[10, 20, 30], [20, 40, 50]])
print(arr)
print(arr.reshape(-1))

# 37

x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
print(type(x))
print(x.shape)
print(x.dtype)

# 38

arr = np.array([[1, 2], [3, 4], [5, 6]])
print(arr)
print(arr.reshape(2, 3))

# 39

arr = np.arange(1, 10)
print(arr)
x = arr.astype(float)
print(x)

# 40

arr = np.full((3, 5), 2, dtype=np.uint)
print(arr)

arr1 = np.ones([3, 5], dtype=np.uint) * 2
print(arr1)

# 41

arr = np.arange(4, dtype=np.int64)

arr1 = np.full_like(arr, 10)
print(arr1)

# 42 diagonal ones 

x = np.eye(3)
print(x)

# 43 diagonal numbers

x = np.diagflat([2, 4, 6, 8, 10])
print(x)

# 44

x = np.arange(0, 51)
print(x)
y = np.arange(10, 51)
print(y)

# 45

arr = np.linspace(2.5, 6.5, 30)
print(arr)

# 46

arr = np.logspace(2., 5., 20, endpoint=False)
print(arr)

# 47

arr = np.tri(4, 3, -1)

print(arr)

# 48

x = np.triu(np.arange(2, 14).reshape(4, 3), -1)
print(x)

# 49

x = np.eye(3)
print(x)

newarr = x.reshape(-1)
print(newarr)

# 50

x = np.array([[2, 4, 6], [6, 8, 10]], np.int32)
print(x)

e = x.flat[4]
print(e)

# 51

x = np.array([[1, 2, 3]])
print(x)

y = np.swapaxes(x, 0, 1)
print(y)
# 52

x = np.zeros((2, 4, 6))
print(x)

print(np.moveaxis(x, 0, -1).shape)

print(np.moveaxis(x, -1, 0).shape)

# 53

x = np.ones((2, 3, 4, 5))
print(x)
print(np.rollaxis(x, 3, 1).shape)

# 54

x = 12.0

print(np.atleast_1d(x))

x = np.arange(6.0).reshape(2, 3)
print(np.atleast_1d(x))

print(np.atleast_1d(1, [3, 4]))

# 55

x = 16
print(np.atleast_1d(x))

arr = np.arange(4.0).reshape(2, 2)
print(np.atleast_1d(arr))

x = 5
print(np.atleast_3d(x))

# 56

arr = np.zeros((3, 4))
print(arr)

arr1 = np.expand_dims(arr, axis=1)
print(arr1)

# 57

x = np.zeros((3, 1, 4))

print(np.squeeze(x))

# 58

a = np.array([[2, 4, 6], [8, 10, 12]])
b = np.array([[1, 3, 5], [7, 9, 11]])

c = np.concatenate((a, b), 1)

print(c)

# 59

a = np.array([10, 20, 30])
b = np.array([40, 50, 60])

c = np.stack((a, b), axis=1)
print(c)

# 60

a = np.array([[10], [20], [30]])
b = np.array([[40], [50], [60]])

c = np.dstack((a, b))

print(c)

# 61

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

arr2 = np.array_split(arr, [2, 6])
print(arr2)

# 62

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

print(np.hsplit(a, [2, 6]))

# 63

x = np.array([[0, 10, 0, 0], [0, -3, 0, 9]])
print(x)

print(np.count_nonzero(x))

# 64

arr = np.zeros((5, 5))
print(arr)

arr += np.arange(5)
print(arr)

# 65 check the presence of an object

x = np.array([[1.12, 2.0, 3.45], [2.33, 5.12, 6.0]], float)

print(1.12 in x)
print(4 in x)

# 66

arr = np.linspace(0, 1, 12, endpoint=True)[1:-1]

print(arr)

# 67

# Importing the NumPy library and aliasing it as 'np'

# Creating a NumPy array 'x' filled with zeros, containing 10 elements
x = np.zeros(10)

# Setting the 'writeable' flag of the array 'x' to False, making it read-only
x.flags.writeable = False

# Printing a message to test whether the array is read-only or not
print("Test the array is read-only or not:")

# Attempting to change the value of the first element in the array 'x'
print("Try to change the value of the first element:")
x[0] = 1  # This line will raise an error since the array is read-only

# 68

arr = np.arange(1, 100)

n = arr[(arr % 3 == 0) | (arr % 5 == 0)]

print(n[:1000])

print(n.sum())


# 69

arr = np.arange(10e3)
print(arr)'''

# 70

x = np.arange(12).reshape(3, 4)

for x in np.nditer(x):
    print(x, end=' ')

print()









