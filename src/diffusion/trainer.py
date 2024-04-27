import torchvision
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

class DiffusionTrainer:
    def __init__(self, time_steps, IMG_SIZE, batch_size, guidance_scale, device="cuda"):
        self.time_steps = time_steps
        self.IMG_SIZE = IMG_SIZE
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.device = device
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        self.vae.to(device)
        
        """Future stuff"""
        self.tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
        )
        self.text_encoder.to(device)
        self.model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True)
        self.model.to(device)

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

    def forward_diffusion_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = DiffusionTrainer.__get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = DiffusionTrainer.__get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # sqrt(alpha_t) x_0 + sqrt(1 - alpha_t) noise
        ret = sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)
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

    def inference(self, prompt):
        print(f"Max len {self.tokenizer.model_max_length}")
        text_input = self.tokenizer(prompt, padding="max_length", max_length = self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input = text_input.input_ids.to(self.device).long()
        uncond_input = self.tokenizer([""] * self.batch_size, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        uncond_input = uncond_input.input_ids.to(self.device).long()
        
        max_val = text_input.max()
        print("text_input max", max_val)
        print("uncond_input max", uncond_input.max())

        exp = torch.tensor([max_val]).long()
        print(exp, exp.float())

        print("Device set to:", self.device)
        print("Input IDs dtype:", text_input.dtype)
        print("Input IDs device:", text_input.device)
        print("Input IDs shape:", text_input.shape)
        print("Input IDs range: min", text_input.min(), "max", text_input.max())

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input)[0]
        uncond_embeddings = self.text_encoder(uncond_input)[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (self.batch_size, self.model.config.in_channels, self.IMG_SIZE // 8, self.IMG_SIZE // 8),
            device=self.device
        )

        for t in tqdm(range(0, self.time_steps)):
            latent_model_input = torch.cat([latents] * 2)

            with torch.no_grad():
                noise_pred = self.model(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = latents - (1/self.time_steps) * noise_pred

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        image.save("output.png")


if __name__ == "__main__":
    time_steps = 300
    trainer = DiffusionTrainer(time_steps, 256, 1, 3)
    trainer.precompute_alphas()
    # trainer.load_dataset()

    trainer.inference(["A car with a red color"])

