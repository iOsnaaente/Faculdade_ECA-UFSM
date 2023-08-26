function interp_4(X,Y)
  
  n = length(X);

  Xn = linspace(X(1), X(n), num=100);
  Yn = zeros(Xn,1);

  for i = 1 : length(Xn)
    Yn(i) = Pn(X,Y,Xn(i));
  endfor

  plot(X,Y,'o', Xn,Yn,'-');

endfunction

function val = Pn(X,Y,Xn)
  n = length(X);
  val = 0;  
  for i = 1 : n
    L = 1;
    for j = 1 : n
      if i != j
        L = L*((Xn-X(j))/(X(i)-X(j)));
      endif
    endfor
    val = val + Y(i)*L;
  endfor
endfunction
