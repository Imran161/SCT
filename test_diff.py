import segmentation_models_pytorch as smp
import torch
import os 
import random
import numpy as np 
import torch
# from utils import set_seed
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

set_seed(0)

device = "cuda:0"
net = smp.FPN(encoder_name="efficientnet-b7",
              in_channels=1+2,
              encoder_weights = "imagenet",
              classes = 2)
net.load_state_dict(torch.load('best_diffusion_model.pth'))
net.to(device)
net.eval()
inp = torch.ones((1, 3, 256, 256), dtype = torch.float).to(device)
inp[:,1:, :, :] = (inp[:,1:, :, :] - torch.tanh(net(inp))+1)/3

x = net(inp).mean()
y = torch.tanh(net(inp)).mean()
print(x, y)
# print("z", z)


# print("z shape", z.shape)
# z_image = z.cpu().detach().numpy()  # Переводим z на CPU и в формат NumPy
z_image = inp[0][1].cpu().detach().numpy()  # Преобразуем z в изображение 1x1

print(inp[0,1,0,0])
# Сохранение изображения
plt.imshow(z_image,# cmap='gray', 
           #vmin=z_image.min(), vmax=z_image.max()
           )
plt.colorbar()
plt.axis('off')
plt.savefig('noisy_masks/z_image.png', bbox_inches='tight')
plt.close()