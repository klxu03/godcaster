import torchvision
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import PNDMScheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

class DiffusionTrainer:
    def __init__(self, time_steps, IMG_SIZE, batch_size, guidance_scale, device="cuda"):
        self.IMG_SIZE = IMG_SIZE
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.device = device

        self.scheduler = PNDMScheduler(beta_schedule="linear")
        self.scheduler.set_timesteps(time_steps)

        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        self.vae.to(device)

        self.tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
        )
        self.text_encoder.to(device)
        
        """Future stuff"""
        self.model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True)
        self.model.to(device)

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
        text_input = self.tokenizer(prompt, padding="max_length", max_length = self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        uncond_input = self.tokenizer([""] * self.batch_size, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device).long())[0]
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device).long())[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (self.batch_size, self.model.config.in_channels, self.IMG_SIZE // 8, self.IMG_SIZE // 8),
            device=self.device
        )

        for t in tqdm(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                noise_pred = self.model(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        image.save("output.png")


if __name__ == "__main__":
    time_steps = 25
    trainer = DiffusionTrainer(time_steps, 512, 1, 7.5)
    # trainer.load_dataset()

    trainer.inference(["A car with a red color"])

