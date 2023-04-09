import _thread 
import socket 
import struct 

HOST = 'localhost'
PORT = 12345

# Número máximo de clientes permitidos 
MAX_CONNECTIONS = 5
# Inicia o arquivo socket 
server = socket.socket() 
# Inicia a escuta no endereço 
server.bind( (HOST, PORT) )
# Define o numero máximo de clientes que podem estar 
#   conectados simultaneamente 
server.listen( MAX_CONNECTIONS ) 
# # Set timeout 
# server.settimeout( 0.1 ) 


# Função que fará a soma 
def calculator( msg ): 
    A, B = msg.split( '+' )
    operators = [ '+', '-', '*', '/' ]
    operations = []
    numbers = []
    aux = ''
    # Varre os caracteres para separar numeros de operadores
    for char in msg:
        # Pega os numeros   
        if char not in operators:
            aux += char 
        # Pega os operadores 
        else:
            numbers.append( float(aux) )
            operations.append( char ) 

    # A quantidade de Numeros deve ser uma unidade menor 
    #   que Operador para que seja possível se calcular 
    calc = 0 
    if len( numbers )-1 == len( operations ): 
        for i in range( len( operations ) ):
            pass
    else: 
        return 0


# Função que irá rodar de forma paralela as solicitações
#   de cada cliente conectado. 
def multi_threaded_client( connection : socket ) -> None:
    while True:
        # Aguarda o recebimento de dados 
        data = connection.recv(2048).decode()
        if data:
            res = calculator( data )
            if res: 
                res = struct.pack( 'f', res )
                connection.send( res )
        else: 
            break


count : int = 0
print( 'Escutando novos clientes...') 
while True: 
    # Aguarda até alguma conexão ser solicitada 
    connection, ( cIP, cPort) = server.accept()
    count += 1
    
    # Inicia uma thread para rodar multi clientes 
    _thread.start_new_thread( multi_threaded_client, ( connection, ) )

    print( 'Connected {} with IP {} '.format( cIP, cPort ) )
    print('Thread Number: ' + str(count) )