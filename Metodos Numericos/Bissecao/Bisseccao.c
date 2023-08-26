/*
Aula Prática 1 - Introdução à Programação

Implemente um algoritmo descrito em linguagem C (preferencialmente) ou Java com o método numérico de busca por raiz de função por bisseção  que:

Permita a escolha da seção inicial (pontos A e B)
Faça o teste para saber se A e B dados são válidos para intervalo da raiz
Permita configuração da tolerância ou seção mínima
Mostre os valores de cada iteração
Mostre o valor encontrado para a raiz com o número de iterações e o erro
Dados os as seguintes informações:

Função: f(x) = x3/10 + x2/2 - 3x + 10

*/

// COMPILADOR -> MinGW32 - gcc

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <locale.h>

// Função: f(x) = x^3/10 + x^2/2 - 3x + 10
double apply(double x){
    return  ((pow(x,3)/10) + (pow(x,2)/2) - 3*x + 10);
}

int main(){
    // Permissão da lingua portuguesa (acentos)
    setlocale(LC_ALL, "portuguese");

    // Declaração dos Pontos A e B
    double Pa, Pb;

    // Iteração e Seção mínima
    int i = 0;
    double secMin = 0.0000001;


    printf("Código de Bisseção de uma função f(x)\n");
    printf("Função-> f(x) = x^3 / 10 + x^2 / 2 - 3x + 10 \n\n");

    printf("Dê entrada aos valores limites de A (inferior) e B (Superior):\n " );

    // %lf -> long float = double
    scanf("%lf", &Pa);
    scanf("%lf", &Pb);


    // Aplicação dos pontos Pa e Pb em F(x)
    double FxA = apply(Pa);
    double FxB = apply(Pb);
    double Pm = (Pa + Pb)/2;
    double Fm = apply(Pm);
    double raiz;

    printf("Os valores de limites são (%2.4lf, %2.4lf) -> (%2.4lf, %2.4lf)\n\n", Pa, Pb, FxA, FxB);

    do{

        FxA = apply(Pa);
        FxB = apply(Pb);
        Fm  = apply(Pm);

        raiz = Pm;

        printf("Iteração %i \t", i);
        printf("Os valores de (%2.4lf, %2.4lf) -> (%2.4lf, %2.4lf) \t", Pa, Pb, FxA, FxB);
        printf("A raiz mais próxima é: %2.4lf\n", raiz);

        if (FxA * FxB > 0){
            printf("\nIntervalo inválido para analise da bissecção, duas raizes entre (%2.4lf , %2.4lf)", Pa, Pb);
            return 0;

        } else if (FxA == 0){
            raiz = Pa;
            break;
        }else if (FxB == 0){
            raiz = Pb;
            break;
        }else{
            if (FxA*Fm > 0)     Pa = Pm;
            else                Pb = Pm;
        }
            Pm = (Pa+Pb)/2;

        printf("--------------------------------------------------------------------------------\n");
        i++;

    }while((Pb-Pa) > secMin);

        printf("\n\n");
        printf("Iteração %i \t", i);
        printf("Os valores de (%2.8lf, %2.8lf) -> (%2.8lf, %2.8lf) \t", Pa, Pb, FxA, FxB);
        printf("\n\nA raiz mais próxima é:  %2.10lf \n\n", Pm);

    return 0;
}
