%f  = input("Entre com a funcao f(x): ", "t");
%x0  = input("Entre com o limite inferior: ");
%x1 = input("Entre com o limite superior: ");
%e  = input("Entre com o erro mahximo tolerado: ");


#secante(f, x0, x1, e)

function Secante_4(f, x0, x1, e)
  
  #Iteracoes
  i = 1;
  
  #Funcao aplicada nos pontos iniciais
  fx0 = f(x0);
  fx1 = f(x1);
  
  #Calculo da primeira iteracao
  x2 = x1 - (f(x1)*(x1-x0))/(f(x1) - f(x0));
  
  #Valor encontrado da Secante
  fx2 = f(x2);
  
  #Calculo do intervalo
  intervalo = abs(x2 - x0); 
  
  #Calculo do erro
  erro = abs(fx2)  
    

  while erro > e && intervalo > e
    
    fx0 = f(x0);
    
    x2 = x1 - (f(x1)*(x1 - x0))/(f(x1) - f(x0));
    
    x0 = x1;
    x1 = x2;
    
    fx2 = f(x2);
    
    fprintf("\n--------------------------------------------------------------------------\n");
    fprintf(" iteracao |  x(i) | x(i+1) |erro 1 = |x(i+1)-x(i)|  |  erro 2 = |f(x(i+1))|\n") ;
    fprintf("----------------------------------------------------------------------------\n");
    fprintf(" %d | x(%d)=%8.10f| x(%d)=%8.10f| |x(%d)-x(%d)|=%8.10f|f(x(%d))=%8.10f|", i, i-1 , x0, i, x2, i, i-1, intervalo, i, abs(fx2));        

    if abs(fx2) < e
      fprintf("O ponto x eh solucao\nA solucao foi obtida atravehs da imagem de f(x)\nO valor da solucao eh x = %8.10f e f(x) = %8.10f", x2,fx2);
      intervalo = linspace(x2-20,x2+20);
      break;
    endif
    
    i++;
    intervalo = abs(x2-x0);
    
    if intervalo < e
      fprintf("O ponto x eh solucao");
      fprintf("\n\nA solucao foi obtida atravehs do domihnio de f(x)");
      fprintf("\n\nO Valor da solucao eh:  x = %8.10f e f(x) = %8.10f", x2 ,fx2);
      break;
    endif   
  endwhile
endfunction