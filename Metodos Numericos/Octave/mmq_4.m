#Definição manual dos valores de entrada
#g = {@(x) x^2 ; @(x) x ; @(x) 1 };

#x = [-1; -0.75; -0.6; -0.5; -0.3; 0; 0.2; 0.4; 0.5; 0.7; 1];
#y = [2.05; 1.153; 0.45; 0.40; 0.5; 0; 0.2; 0.6; 0.512; 1.2; 2.05];

#Function
function mmq_4(x,y,g)
  
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
  
  for j = 1: length(A)
    fprintf("Valores de A(%i): %f \n",j, A(j));
  endfor
  
endfunction

#Chamado da função Mínimos
mmq_4(x, y, g)