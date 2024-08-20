from ultralytics import YOLO
import torch

print("CUDA disponível:", torch.cuda.is_available())
print("Número de GPUs disponíveis:", torch.cuda.device_count())
print("Nome da GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")


# Verifique se CUDA está disponível e defina o dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregue o modelo e mova-o para o dispositivo
model = YOLO("model/yolov10n.pt").to(device)

# Execute a inferência
results = model("teste.jpeg")

# Exiba os resultados
print(results)
