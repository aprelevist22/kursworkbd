//Лабораторная работа №3 - Дифференциальный усилитель
//Вариант 2
//Выполнил студент группы ИСТбд-42 Апрелев Андрей

//Входные данные
//Сопротивление резистора R1, кОм
R1 = 8.2
a = 0.3
b = 0.85
//Расчёт сопротивления резистора R2, учитывая, что R1=aR2 по условию, кОм
R2 = R1 / a
//Расчёт сопротивления резистора R3, учитывая, что R1=bR3 по условию, кОм
R3 = R1 / b

//Решение
//Задание сопротивления переменного резистора R7 как 20 точек от 1 до 20, кОм
R7 = linspace(1, 20, 20)
//Расчёт коэффициента усиления
K = ((R2 + R3) / R1) + 2 * ((R2 * R3) ./ (R1 * R7))
//Вывод графика зависимости коэффициента усиления от сопротивления переменного резистора R7
plot2d(R7, K)
xtitle("","R7","K")
xgrid()
