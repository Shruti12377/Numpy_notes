**Numpy notes**


```python
import numpy as np
```


```python
myarr=np.array([1,3,5,6,7])
print(myarr)
```

    [1 3 5 6 7]
    


```python
myarr.shape
```




    (5,)




```python
myarr.dtype
```




    dtype('int32')



# Array creation:Conversion from other Python structures


```python
listarray=np.array([[1,2,3],[5,8,5],[0,3,1]])
print(listarray)
```

    [[1 2 3]
     [5 8 5]
     [0 3 1]]
    


```python


listarray.size
```




    9




```python
listarray.dtype
```




    dtype('int32')




```python
listarray.shape
```




    (3, 3)




```python
np.array({1,2,4})
```




    array({1, 2, 4}, dtype=object)




```python
zero=np.zeros((2,3))
print(zero)
```

    [[0. 0. 0.]
     [0. 0. 0.]]
    


```python
rng=np.arange(15)
rng
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])




```python
lspace=np.linspace(1,5,12)
print(lspace)
###it will give 12 elements of equal space between 1 and 5.
```

    [1.         1.36363636 1.72727273 2.09090909 2.45454545 2.81818182
     3.18181818 3.54545455 3.90909091 4.27272727 4.63636364 5.        ]
    


```python
emp=np.empty((4,6))
print(emp)
###It will give 4,6 array
```

    [[4.67296746e-307 1.69121096e-306 1.60219442e-306 9.45745515e-308
      7.56587584e-307 1.37961302e-306]
     [1.05699242e-307 8.01097889e-307 1.78020169e-306 7.56601165e-307
      1.02359984e-306 1.33510679e-306]
     [2.22522597e-306 1.33511562e-306 6.23055651e-307 7.56599128e-307
      1.24610587e-306 1.24610723e-306]
     [1.42418172e-306 2.04712906e-306 7.56589622e-307 1.11258277e-307
      8.90111708e-307 2.22522596e-306]]
    


```python
emp_like=np.empty_like(lspace)
print(emp_like)

###it will give an array with size of whatever old array we are passing.
```

    [1.         1.36363636 1.72727273 2.09090909 2.45454545 2.81818182
     3.18181818 3.54545455 3.90909091 4.27272727 4.63636364 5.        ]
    


```python
ide=np.identity(23)
print(ide)

###it will give 23*23 identity matrix
```

    [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
    


```python
ide.shape

```




    (23, 23)




```python
arr=np.arange(99)
print(arr)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
     48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
     72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
     96 97 98]
    


```python
arr.reshape(3,33)

###but we can not give (3,31). because itis not a vlid number for deviding 99 into 3 parts.
```




    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            32],
           [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
            65],
           [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98]])




```python
arr.ravel()
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
           51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
           68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
           85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98])




```python
arr.shape
```




    (99,)



# Numpy Axis
![image.png](attachment:image.png)


```python
x=[[1,2,3],[2,3,4],[3,4,5]]
```


```python
ar=np.array(x)
print(ar)
```

    [[1 2 3]
     [2 3 4]
     [3 4 5]]
    


```python
ar.sum(axis=0)
```




    array([ 6,  9, 12])




```python
ar.sum(axis=1)
```




    array([ 6,  9, 12])




```python
ar.T

###will transpos the original array
```




    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])




```python
ar.flat

###it will flat the array
```




    <numpy.flatiter at 0x19d7f07e430>




```python
for item in ar.flat:
    print(item)
```

    1
    2
    3
    2
    3
    4
    3
    4
    5
    


```python
for item in ar:
    print(item)
    
###show the difference between ar and ar.flat in output
```

    [1 2 3]
    [2 3 4]
    [3 4 5]
    


```python
ar.ndim

###will return number of dimension
```




    2




```python
ar.size
```




    9




```python
ar.nbytes
```




    36




```python
one=np.array([1,24,6,5,7])
print(one)
```

    [ 1 24  6  5  7]
    


```python
one.argmax()

### will return index of maximum element
```




    1




```python
one.argmin()
```




    0




```python
one.argsort()

###will sort the element by index
```




    array([0, 3, 2, 4, 1], dtype=int64)




```python
###let's take another array

x=[[2,3,6],[4,6,2],[2,5,1]]

a=np.array(x)
print(a)
```

    [[2 3 6]
     [4 6 2]
     [2 5 1]]
    


```python
a.argmax()
```




    2




```python
a.argmax(axis=0)
```




    array([1, 1, 0], dtype=int64)




```python
a.argmin(axis=1)
```




    array([0, 2, 2], dtype=int64)




```python
a.argsort(axis=0)
```




    array([[0, 0, 2],
           [2, 2, 1],
           [1, 1, 0]], dtype=int64)




```python
a.ravel()
```




    array([2, 3, 6, 4, 6, 2, 2, 5, 1])




```python
a
```




    array([[2, 3, 6],
           [4, 6, 2],
           [2, 5, 1]])




```python
ar
```




    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])




```python
a+ar

### matrix calculations(operations)
```




    array([[3, 5, 9],
           [6, 9, 6],
           [5, 9, 6]])




```python
a*ar
```




    array([[ 2,  6, 18],
           [ 8, 18,  8],
           [ 6, 20,  5]])




```python
np.sqrt(a*ar)
```




    array([[1.41421356, 2.44948974, 4.24264069],
           [2.82842712, 4.24264069, 2.82842712],
           [2.44948974, 4.47213595, 2.23606798]])




```python
np.sqrt(a)
```




    array([[1.41421356, 1.73205081, 2.44948974],
           [2.        , 2.44948974, 1.41421356],
           [1.41421356, 2.23606798, 1.        ]])




```python
a.sum()
```




    31




```python
a.min()
```




    1




```python
a.max()
```




    6




```python
ar
```




    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])




```python
np.where(ar>1)
```




    (array([0, 0, 1, 1, 1, 2, 2, 2], dtype=int64),
     array([1, 2, 0, 1, 2, 0, 1, 2], dtype=int64))




```python
np.nonzero(a)
```




    (array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=int64),
     array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=int64))




```python
import sys
```


```python
py_ar=[2,4,6,9]
```


```python
np_ar=np.array(py_ar)
```


```python
sys.getsizeof(1)*len(py_ar)
```




    112




```python
np_ar.itemsize*np_ar.size

### that's why we use numpy array 
```




    16




```python

```
