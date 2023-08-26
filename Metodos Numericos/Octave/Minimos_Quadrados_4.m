#Definição manual dos valores de entrada
g = {@(x) x^2 ; @(x) x ; @(x) 1 };

x = [-1; -0.75; -0.6; -0.5; -0.3; 0; 0.2; 0.4; 0.5; 0.7; 1];
y = [2.05; 1.153; 0.45; 0.40; 0.5; 0; 0.2; 0.6; 0.512; 1.2; 2.05];

#Dar entrada a um valor
xt = 0.6;

minimos(x, y, g, xt)


function minimos(x,y,g,xt)
  
  #Calculo de A
  A = zeros( length(g), "float");

  for i = 1: length(g)
    for j = 1:length(g)
      A(i,j) = 0;
      for k = 1:length(x)
          A(i,j) = A(i,j) + g{i}(x(k))*g{j}(x(k));
        endfor
    endfor  
  endfor

  #Calculo de b
  b = zeros(length(g));

  for i=1:length(g)
    b(i) = 0;
    for k=1:length(x)
      b(i) = b(i) + y(k)*g{i}(x(k));
    endfor
  endfor

  #Solucao A.a = b [Dimensao n.1]
  a = inv(A)*b;
  
  #Calculo de Ytio -> Yt = fi(Xtio)
  yt = 0;
  
  for k=1:length(g)
    yt = yt + a(k)*g{k}(xt);
  endfor
  
  fprintf("%f\n", yt)
  
endfunction