from random import randint as rand
import numpy as num
import matplotlib.pyplot as plt

def read_dat():
    try:
        print("Введите размер матрицы")
        n = int(input())
        if (n % 2) | (n < 1):
            raise Exception("Размер должен быть чётным")
        print("Введите множитель")
        k = int(input())

        a = num.array([[rand(-10, 10) for j in range(n)] for i in range(n)])
        print("Матрица А")
        print(a)

        e = a[0: n // 2, 0: n // 2]
        b = a[0: n // 2, n // 2:]
        d = a[n // 2:, 0: n // 2]
        c = a[n // 2:, n // 2:]
    except ValueError:
        print("Ваше значение не число")
        exit(0)
    except Exception as exc:
        print(exc)
        print("Перезапустите программу")
        exit(0)
    return [e,b,d,c,a,k,n]

def matrix_change(a,e,b,d,c):
    try:
        f = a.copy()

        poselem = num.count_nonzero(c[:,1::2]>0)
        print("Чётные столбцы матрицы C")
        print(c[:,1::2])
        print("Число положительных ненулевых элементов в чётных столбцах")
        print(poselem)

        negelem = num.count_nonzero(c[:, ::2] < 0)
        print("Нечётные столбцы матрицы C")
        print(c[:, ::2])
        print("Число отрицательных элементов в нечётных столбцах")
        print(negelem)

        if (poselem > negelem):
            b = num.flipud(a[n // 2:, n // 2:])
            c = num.flipud(a[0: n // 2, n // 2:])
        else:
            e = a[n // 2:, 0: n // 2]
            c = a[0: n // 2, 0: n // 2]

        print("Новые значения подматриц")
        print("e")
        print(e)
        print("b")
        print(b)
        print("d")
        print(d)
        print("c")
        print(c)

        f = num.concatenate((num.concatenate((e, d), axis=0), num.concatenate((b, c), axis=0)), axis=1);

        print("Матрица F")
        print(f)
    except Exception as exc:
        print(exc)
        print("Перезапустите программу")
        exit(0)

    return [e,b,d,c,f]

def calculate(a,f,k):#
    try:
        print("Определитель матрицы A")
        det = num.linalg.det(a)
        print(det)

        print("Диагональная сумма матрицы  F")
        dia = num.trace(f)
        print(dia)

        if det > dia:
            at = num.transpose(a)
            ainv = num.linalg.inv(a)
            fainv=num.multiply(f,ainv)
            result = num.subtract(num.multiply(a, at), num.multiply(k, fainv))
        else:
            ainv = num.linalg.inv(a)
            g = num.tril(a)
            ft = num.transpose(f)
            result = num.multiply(num.subtract(num.add(num.multiply(k, ainv), g),ft) , k)

        print("Результат")
        print(result)
    except Exception as exc:
        print(exc)
        print("Перезапустите программу")
        exit(0)

def graphs(e,b,d,c):
    try:
        graph1 = plt.figure()
        esize = len(e) * len(e[0])
        x = num.linspace(0, esize, esize)
        y = num.asarray(e).reshape(-1)
        plt.plot(x, y)
        plt.title("Матрица E")
        plt.show()

        graph2 = plt.figure()
        bsize = len(b) * len(b[0])
        x = num.linspace(0, bsize, bsize)
        y = num.asarray(b).reshape(-1)
        plt.plot(x, y)
        plt.title("Матрица B")
        plt.show()

        graph3 = plt.figure()
        dsize = len(d) * len(d[0])
        x = num.linspace(0, dsize, dsize)
        y = num.asarray(d).reshape(-1)
        plt.plot(x, y)
        plt.title("Матрица D")
        plt.show()

        graph4 = plt.figure()
        csize = len(c) * len(c[0])
        x = num.linspace(0, csize, csize)
        y = num.asarray(c).reshape(-1)
        plt.plot(x, y)
        plt.title("Матрица C")
        plt.show()
    except Exception as exc:
        print(exc)
        print("Перезапустите программу")
        exit(0)

if __name__ == '__main__':
    print("Вариант 2")

    try:
        values=read_dat()
        e=values[0]
        b=values[1]
        d=values[2]
        c=values[3]
        a=values[4]
        k=values[5]
        n=values[6]

        print("Подматрица e")
        print(e)
        print("Подматрица b")
        print(b)
        print("Подматрица d")
        print(d)
        print("Подматрица c")
        print(c)

        res=matrix_change(a,e,b,d,c)
        e = res[0]
        b = res[1]
        d = res[2]
        c = res[3]
        f = res[4]


        calculate(a,f,k)

        graphs(e,b,d,c)

    except ValueError:
        print("Ваше значение не число")
        exit(0)

    except Exception as exc:
        print(exc)
        print("Перезапустите программу")
        exit(0)
