A  = input("Entre com a Matriz A(n.n): ");
B  = input("Entre com a Matriz B(n.1): ");
Ap = input("Entre com a aproximacao inicial x^0(n.1): ");
e  = input("Entre com o erro mahximo tolerado: ");

#Gauss_Jacobi(A, B, Ap, e)

#Definicao da funcao que encontra as raizes reais da funcao
#function Gauss_Jacobi(A, B, Ap, e)
  
  C = zeros(size(A));
  g = zeros(length(B),1);
  
  
  #Construir as matrizes C e G 
  for i = 1:length(A)
    g(i) = B(i)/A(i,i);
    for j = 1:length(A)
      if i != j
          C(i,j) = -A(i,j)/A(i,i);
        endif
      endfor
  endfor

  #Testa a condicao de aplicacao do teste
  if norm(C,1) < 1
    #Nuhmero de iteracao
    n=0;
    erro = 1;
    
    while erro > e
      An = C*Ap +g;
      erro = norm(An - Ap)/norm(An);
      Ap = An;
      n++;
      
    endwhile
    
    #Print da solução  
    fprintf("A solucao eh a matriz: \n")
    for i=1:length(Ap)
      fprintf("%f ", Ap(i,:))
      fprintf("\n")
    endfor
  
  else
    fprintf("Nao hah convergencia!!")
  endif 
  
#endfunction