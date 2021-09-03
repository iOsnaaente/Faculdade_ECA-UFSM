package sotr4;

import java.util.Scanner;
import java.lang.Math;

/**
 * @author Bruno Sampaio
 * 
 * f(x) = 0.66 x5 - 8.7 x4 + 42 x3 - 90 x2 + 80 x - 18
 *  
 *  Desafio:
 *   Instancie um objeto com a classe do método de Newton-Raphson
 *   Encontrar todas as raízes da função
 */

public class NewtonRasp {
    
    // Numero de intervalos que será percorrido para a verificação de raizes 
    public int minimos = 100;
    public double pedacinho;
    
    // Limites entre os intervalos
    public double pA,pB;
    
    // Ponto médio = chute 
    public double pM;
    
    // Definições de parada
    public double tolerancia = 0.000001;

    // Numero máximo de iteracoes
    public int iteracaoMax = 20;

    // Array para guardar as raizes
    public int numeroRaizes = 5;
    public double raizes[] = new double[numeroRaizes];
    
    // Funcao de aplicação das funcoes
    public static double func(double x){
        // Aplicação de x na função:
        return (0.66*Math.pow(x,5) - 8.7*Math.pow(x,4) +42*Math.pow(x,3) -90*Math.pow(x,2) + 80*x - 18);
    }
    
    // Funcao da aplicacao da derivada da funcao
    public static double dfunc(double x){
        // Para calcular a derivada numérica de f(x) usa-se:
        // f'(x) = ( f(x + erro) - f(x) ) / erro 
        double erro = 0.00001;
        return (( func(x + erro) - func(x) ) / erro) ; 
    }
  
    // Metodo que calcula cada raiz dentro dos limites [Pa, Pb]
    public double NR_raiz(double Pa, double Pb){
       
        double x0 = (Pb + Pa)/2 ; 
        
        double fx0 = func(x0);
        double dfx0 = dfunc(x0); 
        
        // Primeira iteração do método
        double z = x0 - (fx0 / dfx0);
        
        if (z < pA  || z > pB){
            return 0xffffffff;
        }
       
        int i = 1;
        
        double fz = func(z);
        
        double intervalo = z - x0;
 
        double tol = Math.abs(fz);
        
        System.out.printf("\niteracao |  x(i)           | x(i+1)            |erro 1 = |x(i+1)-x(i)|  |  erro 2 = |f(x(i+1))|                          ") ;    
      
        while (intervalo > tolerancia ){
            
            if ( Math.abs(z - x0) < tolerancia){
                System.out.printf("\n\nO ponto x = %2.10f é solução", x0);
                System.out.printf("\nA solução foi obtida através do domínio de f(x)");
                System.out.printf("\n\nO Valor da solução é:  x = %8.6f e f(x) = %8.6f", x0,fx0);
                break;
            
            }else if (tol < tolerancia){
                System.out.printf("\n\nO ponto x = %2.10f é solução", x0);
                System.out.printf("\nA solução foi obtida através da imagem de f(x)");
                System.out.printf("\n\nO Valor da solução é:  x = %8.6f e f(x) = %8.6f", x0,fx0);
                break;
                
            } else if ( i > iteracaoMax){
                System.out.printf("\n\nO ponto x = %2.10f é solução", x0);
                System.out.printf("\nA solução foi obtida através do número máximo de iterações iteracaoMax = %d", iteracaoMax);
                System.out.printf("\n\nO Valor da solução é:  x = %8.6f e f(x) = %8.6f", x0,fx0);
                break;
            }
            
            x0 = z ;
            fx0 = func(x0);
            dfx0 = dfunc(x0);
            
            z = x0 - (fx0 / dfx0);
            
            fz = func(z);
            tol = Math.abs(fz);
                        
            System.out.printf("\n-----------------------------------------------------------------------------------------------------------------------------");
            System.out.printf("\n   %d      | x(%d)=%10.8f | x(%d)=%10.8f   | |x(%d)-x(%d)|=%10.8f | f(x(%d))=%10.8f  |", i, i-1, x0, i, z, i, i-1, intervalo, i, Math.abs(fx0));
           
            i = i + 1 ;
        }
        return x0;
    }
    
    // Método construtor ??
    public void NewtonRasp(){
        
        Scanner in = new Scanner (System.in);
        
        System.out.print("Digite o valor do limite inferior: ");
        pA = in.nextDouble();
        System.out.print("Digite o valor do limite superior: ");
        pB = in.nextDouble();
        
        pedacinho = (pB - pA) / minimos;
        
        double espacinho = pA;
        double teste;
        int ident = 0;
        
        for ( int i = 1; i<100; i++){
            teste = NR_raiz(espacinho, (espacinho = espacinho+pedacinho));
            if (teste != 0xffffffff){
                raizes[ident] = teste;
                ident = ident + 1;
            }
        }
        
        System.out.printf("\n\nAs raizes encontradas foram:");
        for (int i = 0; i<numeroRaizes; i++){
            System.out.printf("\nx=%10.10f em f(x)=%10.10f", raizes[i], func(raizes[i]));
        }
    }
}
