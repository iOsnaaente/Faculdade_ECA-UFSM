import socket 

HOST = 'localhost'
PORT = 12345

# Inicia o arquivo socket 
client = socket.socket() 

# Solicita conexão 
client.connect( (HOST, PORT) )


while True:
    # Input deve ser dois Números A B separados por um backspace 
    msg = input( 'Entre com o valor de A e B: ')
    # Se a mensagem for '' ele encerra 
    if msg == '':
        break
    # Se houver mensagem 
    else: 
        # Envia para o servidor resolver a soma 
        client.send( msg.encode() )
        
        # Recebe o resultado da soma em ans
        ans = client.recv( 1024 ).decode() 
        A, B = msg.split( ' ' )

        print( 'O resultado de {} + {} é {}\n'.format( A, B, ans ) )