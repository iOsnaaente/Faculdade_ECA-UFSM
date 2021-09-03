f = input("Entre com a função f(x): ");
A = input("Entre com o limite inferior A: ");
B = input("Entre com o limite superior B: ");
e = input("Entre com o erro máximo permitido: ");

bissec_4(f, A, B, e)

#Definição da função que encontra as raizes reais da função
function bissec_4(f, A, B, e)

  #Valor de A e B na funcao
  fA = f(A);
  fB = f(B);

  i = 0;
  if fA*fB > 0
    fprintf("Não existe raiz no intervalo [%f,%f] ou o intervalo é muito grande", A, B);
  else
    
    #Definição do intervalo para parada posterior  
    int = abs(B-A);

    #Ponto Médio
    Pm = (A+B)/2;

    while int > e 

      if f(Pm) == 0  
        break;
      endif
      
      #Condições de zero em um intervalo 
      if f(A) * f(Pm) > 0 
        A = Pm;  
      else
        B = Pm;
      endif   
      Pm = (B+A)/2;
      
      i++;
      fprintf("Iteração %i com fPm = %f\n", i, f(Pm));
      int = abs(B-A);
    endwhile
    
    fprintf("O zero da equação esta próximo do ponto x=%f onde f(%f)=%f", Pm, Pm, f(Pm));
  endif

endfunction