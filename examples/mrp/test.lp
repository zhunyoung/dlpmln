1{nn_edge(g,0,t); nn_edge(g,0,f)}1.
1{nn_edge(g,1,t); nn_edge(g,1,f)}1.
1{nn_edge(g,2,t); nn_edge(g,2,f)}1.
1{nn_edge(g,3,t); nn_edge(g,3,f)}1.
1{nn_edge(g,4,t); nn_edge(g,4,f)}1.
1{nn_edge(g,5,t); nn_edge(g,5,f)}1.
1{nn_edge(g,6,t); nn_edge(g,6,f)}1.
1{nn_edge(g,7,t); nn_edge(g,7,f)}1.
1{nn_edge(g,8,t); nn_edge(g,8,f)}1.
1{nn_edge(g,9,t); nn_edge(g,9,f)}1.
1{nn_edge(g,10,t); nn_edge(g,10,f)}1.
1{nn_edge(g,11,t); nn_edge(g,11,f)}1.
1{nn_edge(g,12,t); nn_edge(g,12,f)}1.
1{nn_edge(g,13,t); nn_edge(g,13,f)}1.
1{nn_edge(g,14,t); nn_edge(g,14,f)}1.
1{nn_edge(g,15,t); nn_edge(g,15,f)}1.
1{nn_edge(g,16,t); nn_edge(g,16,f)}1.
1{nn_edge(g,17,t); nn_edge(g,17,f)}1.
1{nn_edge(g,18,t); nn_edge(g,18,f)}1.
1{nn_edge(g,19,t); nn_edge(g,19,f)}1.
1{nn_edge(g,20,t); nn_edge(g,20,f)}1.
1{nn_edge(g,21,t); nn_edge(g,21,f)}1.
1{nn_edge(g,22,t); nn_edge(g,22,f)}1.
1{nn_edge(g,23,t); nn_edge(g,23,f)}1.
grid(g).
nn_edge(g, X):- nn_edge(g,X,t).
sp(0,1) :- nn_edge(g, 0).
sp(1,2) :- nn_edge(g, 1).
sp(2,3) :- nn_edge(g, 2).
sp(4,5) :- nn_edge(g, 3).
sp(5,6) :- nn_edge(g, 4).
sp(6,7) :- nn_edge(g, 5).
sp(8,9) :- nn_edge(g, 6).
sp(9,10) :- nn_edge(g, 7).
sp(10,11) :- nn_edge(g, 8).
sp(12,13) :- nn_edge(g, 9).
sp(13,14) :- nn_edge(g, 10).
sp(14,15) :- nn_edge(g, 11).
sp(0,4) :- nn_edge(g, 12).
sp(4,8) :- nn_edge(g, 13).
sp(8,12) :- nn_edge(g, 14).
sp(1,5) :- nn_edge(g, 15).
sp(5,9) :- nn_edge(g, 16).
sp(9,13) :- nn_edge(g, 17).
sp(2,6) :- nn_edge(g, 18).
sp(6,10) :- nn_edge(g, 19).
sp(10,14) :- nn_edge(g, 20).
sp(3,7) :- nn_edge(g, 21).
sp(7,11) :- nn_edge(g, 22).
sp(11,15) :- nn_edge(g, 23).
sp(X,Y) :- sp(Y,X).
mistake :- X=0..15, #count{Y: sp(X,Y)} = 1.
mistake :- X=0..15, #count{Y: sp(X,Y)} >= 3.
reachable(X, Y) :- sp(X, Y).
reachable(X, Y) :- reachable(X, Z), sp(Z, Y).
mistake :- sp(X, _), sp(Y, _), not reachable(X, Y).

:- mistake.
sp(external, 0).
sp(external, 11).
:~ nn_edge(0). [6,0]
:~ nn_edge(1). [2,1]
:~ nn_edge(2). [6,2]
:~ nn_edge(3). [2,3]
:~ nn_edge(4). [6,4]
:~ nn_edge(5). [6,5]
:~ nn_edge(6). [2,6]
:~ nn_edge(7). [6,7]
:~ nn_edge(8). [6,8]
:~ nn_edge(9). [2,9]
:~ nn_edge(10). [2,10]
:~ nn_edge(11). [6,11]
:~ nn_edge(12). [6,12]
:~ nn_edge(13). [2,13]
:~ nn_edge(14). [6,14]
:~ nn_edge(15). [6,15]
:~ nn_edge(16). [6,16]
:~ nn_edge(17). [6,17]
:~ nn_edge(18). [6,18]
:~ nn_edge(19). [2,19]
:~ nn_edge(20). [6,20]
:~ nn_edge(21). [2,21]
:~ nn_edge(22). [6,22]
:~ nn_edge(23). [2,23]
