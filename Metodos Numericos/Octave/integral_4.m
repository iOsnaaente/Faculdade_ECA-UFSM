# Nome = "Bruno Gabriel Flores Sampaio"
# Matricula = 2017*****

#REGRA DOS TRAPEHZIOS REPETIDOS

function integral_4(f,a,b,e)
  
  ##Chama as funções de trapezio e Simpson
  IntegralTrapezio(f,a,b,e);
  IntegralSimpson(f,a,b,e);
  
endfunction


##METODO DOS TRAPEZIOS REPETIDOS 
function IntegralTrapezio(f,a,b,e)
 
#Distancia entre os intervalos
d = 0.001;

#Vetor de m pontos cobrindo o intervalo [a,b] 
xd = linspace(a,b,(b-a)/d);

for i = 1 : length(xd)
  yd =( f(xd(i)+2*d) - 2*f(xd(i)+d) +  f(xd(i)) )/(d^2);
endfor

#Valor máximo da derivada segunda dos subintervalos 
M = max(abs(yd));

m = sqrt((((b-a)^3)*M)/(12*e));

##Arredondar o numero para cima
if (m>round(m)) 
  m = round(m)+1;
else
  m = round(m);  
endif

#CALCULO DA INTEGRAL 
h = (b-a)/m;
s = f(a) + f(b);
x = a;

for k = 1 : (m - 1)
  x = x + h;
  s = s + 2*f(x);
endfor 

#Valor aproximado da integral que sera devolvido
Itr = (h/2)*s;

fprintf("O valor da integral pelo método dos trapézios repetidos é: %8.10f \n", Itr);

endfunction


#MÉTODO DA REGRA 1/3 DE SIMPSON REPETIDA 
function IntegralSimpson(f,a,b,e)
  
  #Um numero muito pequeno
  d = 0.001;

  #Vetor de m pontos cobrindo o intervalo [a,b] 
  xd = linspace(a,b,(b-a)/d);

  for i = 1 : length(xd)
    yd =(  f(xd(i)+4*d) - 4*f(xd(i)+3*d) + 6*f(xd(i)+2*d) - 4*f(xd(i)+d) +  f(xd(i)) )/(d^4);
  endfor

  #Valor máximo da derivada segunda dos subintervalos 
  M = max(abs(yd));

  #Calculo de m   
  m = ((((b-a)^5)*M)/(180*e))^(1/4);

  #O arredondamento deve ser feito para cima
  if (m>round(m)) 
    m = round(m)+1;
  else
    m = round(m);  
  endif
  
  #Deve-se tomar o primeiro número par, maior que m 
  if (rem(m,2)==1) 
    m = m+1;
  endif 
  
  h = (b-a)/(m);
  s = f(a) + f(b);
  
  x = a +h;
  
  for k=1 : 2 : (m-1)
    s = s + 4*f(x);
    x = x + 2*h;
  endfor

  x = a +2*h;
    
  for k=2 : 2: (m-2)
    s = s + 2*f(x);
    x = x + 2*h;
  endfor
  
  Isr = (h/3)*s;
  
  fprintf("O valor da integral pelo método de 1/3 de Simpson repetido é: %8.10f \n", Isr);

endfunction