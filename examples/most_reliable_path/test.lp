grid(g).
nn(m(g,24), nn_edge, [t,f]) :- grid(g).
sp(0,1) :- nn_edge(g, 0, t).
sp(1,2) :- nn_edge(g, 1, t).
sp(2,3) :- nn_edge(g, 2, t).
sp(4,5) :- nn_edge(g, 3, t).
sp(5,6) :- nn_edge(g, 4, t).
sp(6,7) :- nn_edge(g, 5, t).
sp(8,9) :- nn_edge(g, 6, t).
sp(9,10) :- nn_edge(g, 7, t).
sp(10,11) :- nn_edge(g, 8, t).
sp(12,13) :- nn_edge(g, 9, t).
sp(13,14) :- nn_edge(g, 10, t).
sp(14,15) :- nn_edge(g, 11, t).
sp(0,4) :- nn_edge(g, 12, t).
sp(4,8) :- nn_edge(g, 13, t).
sp(8,12) :- nn_edge(g, 14, t).
sp(1,5) :- nn_edge(g, 15, t).
sp(5,9) :- nn_edge(g, 16, t).
sp(9,13) :- nn_edge(g, 17, t).
sp(2,6) :- nn_edge(g, 18, t).
sp(6,10) :- nn_edge(g, 19, t).
sp(10,14) :- nn_edge(g, 20, t).
sp(3,7) :- nn_edge(g, 21, t).
sp(7,11) :- nn_edge(g, 22, t).
sp(11,15) :- nn_edge(g, 23, t).

sp(X,Y) :- sp(Y,X).
mistake :- X=0..15, #count{Y: sp(X,Y)} = 1.
mistake :- X=0..15, #count{Y: sp(X,Y)} >= 3.
reachable(X, Y) :- sp(X, Y).
reachable(X, Y) :- reachable(X, Z), sp(Z, Y).
mistake :- sp(X, _), sp(Y, _), not reachable(X, Y).

1{nn_edge(g, 0, t); nn_edge(g, 0, f)}1.
1{nn_edge(g, 1, t); nn_edge(g, 1, f)}1.
1{nn_edge(g, 2, t); nn_edge(g, 2, f)}1.
1{nn_edge(g, 3, t); nn_edge(g, 3, f)}1.
1{nn_edge(g, 4, t); nn_edge(g, 4, f)}1.
1{nn_edge(g, 5, t); nn_edge(g, 5, f)}1.
1{nn_edge(g, 6, t); nn_edge(g, 6, f)}1.
1{nn_edge(g, 7, t); nn_edge(g, 7, f)}1.
1{nn_edge(g, 8, t); nn_edge(g, 8, f)}1.
1{nn_edge(g, 9, t); nn_edge(g, 9, f)}1.
1{nn_edge(g, 10, t); nn_edge(g, 10, f)}1.
1{nn_edge(g, 11, t); nn_edge(g, 11, f)}1.
1{nn_edge(g, 12, t); nn_edge(g, 12, f)}1.
1{nn_edge(g, 13, t); nn_edge(g, 13, f)}1.
1{nn_edge(g, 14, t); nn_edge(g, 14, f)}1.
1{nn_edge(g, 15, t); nn_edge(g, 15, f)}1.
1{nn_edge(g, 16, t); nn_edge(g, 16, f)}1.
1{nn_edge(g, 17, t); nn_edge(g, 17, f)}1.
1{nn_edge(g, 18, t); nn_edge(g, 18, f)}1.
1{nn_edge(g, 19, t); nn_edge(g, 19, f)}1.
1{nn_edge(g, 20, t); nn_edge(g, 20, f)}1.
1{nn_edge(g, 21, t); nn_edge(g, 21, f)}1.
1{nn_edge(g, 22, t); nn_edge(g, 22, f)}1.
1{nn_edge(g, 23, t); nn_edge(g, 23, f)}1.

:- mistake.
sp(external, 5).
sp(external, 11).
:~ nn_edge(g, 0, t). [2,0]
:~ nn_edge(g, 1, t). [2,1]
:~ nn_edge(g, 2, t). [2,2]
:~ nn_edge(g, 3, t). [2,3]
:~ nn_edge(g, 4, t). [2,4]
:~ nn_edge(g, 5, t). [2,5]
:~ nn_edge(g, 6, t). [6,6]
:~ nn_edge(g, 7, t). [2,7]
:~ nn_edge(g, 8, t). [2,8]
:~ nn_edge(g, 9, t). [2,9]
:~ nn_edge(g, 10, t). [6,10]
:~ nn_edge(g, 11, t). [6,11]
:~ nn_edge(g, 12, t). [2,12]
:~ nn_edge(g, 13, t). [6,13]
:~ nn_edge(g, 14, t). [2,14]
:~ nn_edge(g, 15, t). [2,15]
:~ nn_edge(g, 16, t). [2,16]
:~ nn_edge(g, 17, t). [2,17]
:~ nn_edge(g, 18, t). [2,18]
:~ nn_edge(g, 19, t). [2,19]
:~ nn_edge(g, 20, t). [2,20]
:~ nn_edge(g, 21, t). [2,21]
:~ nn_edge(g, 22, t). [2,22]
:~ nn_edge(g, 23, t). [6,23]
