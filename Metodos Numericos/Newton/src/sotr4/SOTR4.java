package sotr4;

import java.util.Scanner;
import java.lang.Math;

/**
 * @author Bruno Sampaio
 * 
 * f(x) = 0.66 x5 - 8.7 x4 + 42 x3 - 90 x2 + 80 x - 18
 *  
 */

/*
    Requisitos da implementação:

    OK Permitir a escolha do ponto inicial
    OK Mostrar os valores de cada iteração
    
    Mostrar o valor encontrado para a raiz com o número de iterações e o erro

    Desafio:
    Instancie um objeto com a classe do método de Newton-Raphson
    Encontrar todas as raízes da função
*/

public class SOTR4 {

    public int minimos = 100;
   
    public double pA,pB;
    public double pM;
        
    public static double func(double x){
        // Aplicação de x na função:
        return (0.66*Math.pow(x,5) - 8.7*Math.pow(x,4) +42*Math.pow(x,3) -90*Math.pow(x,2) + 80*x - 18);
    }
    
    public static double dfunc(double x){
        // Para calcular a derivada numérica de f(x) usa-se:
        // f'(x) = ( f(x + erro) - f(x) ) / erro 
        double erro = 0.00001;
        return (( func(x + erro) - func(x) ) / erro) ; 
    }
    
    public static void main(String[] args) {
        
        // Definições de parada
        double tolerancia = 0.000001;
        
        // Numero máximo de iteracoes
        int iteracaoMax = 20;
        
        // Iteração 
        int i = 1 ;
        
        // Entrada em um número - chute qualquer
        System.out.print("Digite o valor inicial: ");
        Scanner in = new Scanner (System.in);
        
        // Valor do chute 
        double x0 = in.nextDouble();
        
        double fx0 = func(x0);
        double dfx0 = dfunc(x0); 
        
        // Primeira iteração do método
        double z = x0 - (fx0 / dfx0);
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
    }
}
