import cv2

# Função de callback para capturar eventos do mouse
def capturar_coordenadas(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Quando o botão esquerdo do mouse é clicado
        print(f"Coordenadas: x={x}, y={y}")

# Abrir o vídeo
video_path = 'sample\Sábado 30-11-2024.mp4'
cap = cv2.VideoCapture(video_path)

# Configurar o evento do mouse
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', capturar_coordenadas)

# Loop para exibir o vídeo
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)
    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
