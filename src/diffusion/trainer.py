import torchvision
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DiffusionTrainer:
    def __init__(self, time_steps, IMG_SIZE, batch_size):
        self.time_steps = time_steps
        self.IMG_SIZE = IMG_SIZE
        self.batch_size = batch_size
        pass

    def __beta_schedule(time_steps=1000, start=0.001, end=0.02):
        schedule = torch.linspace(start, end, time_steps)
        return schedule
    
    def precompute_alphas(self):
        self.betas = DiffusionTrainer.__beta_schedule(time_steps=self.time_steps)

        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    """
    Helper function to generate noise for forward diffusion process at timestep t given noise alpha levels vals

    Args:
        vals (list): cumulative product immediate calculation of noise from timestep t
        t (torch.Tensor): Tensor of indices of timesteps
        x_shape (tuple): Shape of the input tensor
    """
    def __get_index_from_list(vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cuda"):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = DiffusionTrainer.__get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = DiffusionTrainer.__get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # sqrt(alpha_t) x_0 + sqrt(1 - alpha_t) noise
        ret = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
        return ret
        
    def load_dataset(self):
        data_transforms = [
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # Scales data into [0,1] 
            transforms.Lambda(lambda t: (t * 2) - 1) # Finally scales between [-1, 1] 
        ]

        data_transforms = transforms.Compose(data_transforms)

        train = torchvision.datasets.StanfordCars(root="./", transform=data_transforms, download=True)
        test = torchvision.datasets.StanfordCars(root="./", transform=data_transforms, download=True, split="test")

        data = torch.utils.data.ConcatDataset([train, test])
        self.dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return data

    def show_tensor_image(file_name, image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        # Take first image of batch
        if len(image.shape) == 4:
            image = image[0, :, :, :] 
        plt.imshow(reverse_transforms(image))
        plt.imsave(file_name, reverse_transforms(image))

if __name__ == "__main__":
    time_steps = 300
    trainer = DiffusionTrainer(time_steps, 64, 128)
    trainer.precompute_alphas()
    trainer.load_dataset()

    # Simulate forward diffusion
    image = next(iter(trainer.dataloader))[0]

    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(time_steps/num_images)

    for idx in range(0, time_steps, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
        img, noise = trainer.forward_diffusion_sample(image, t)
        DiffusionTrainer.show_tensor_image(f"noise_schedule.png", img.cpu())

    plt.savefig('diffusion_process.png', dpi=300)  # Saves as PNG with high resolution
    plt.close()  # Close the plotting window to free up system resources