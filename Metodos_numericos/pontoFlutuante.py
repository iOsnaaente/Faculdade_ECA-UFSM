# Criado por Bruno Gabriel Flores Sampaio em 30/12/2020
# Universidade Federal de Santa Maria 
# Matricula 201720094
# Disciplina MTM 224 – Métodos Numéricos Computacionais – Turma 12 (em REDE)
# Professor Alex Andre Schimidt 

mantissa = 16 

# decomposição do numero 44.71 em inteiro e decimal 
numInt = "1"
numDec = "000000000000001"

soma = len(numInt)+len(numDec)

if len(numInt) < mantissa:
    if soma > mantissa: 
        dig = mantissa - len(numInt)
        newDec = ""
        for i in range(dig):
            newDec = newDec+'1' if numDec[i] == '1' else newDec+'0'
        numDec = newDec
    
    numFloat = 0
    for i in range(1, len(numInt)+1, 1):
        num = 1 if numInt[-i] == '1' else 0 
        numFloat = numFloat + num*(2**(i-1))
        
    for i in range(1, len(numDec)+1, 1):
        num = 1 if numDec[i-1] == '1' else 0 
        numFloat = numFloat + num/(2**(i))
    
    print(numFloat)

numReal = 1

print("Diferença = ", numReal-numFloat)
