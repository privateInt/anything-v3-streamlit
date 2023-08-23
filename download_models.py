from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
from utils import gen_models, gen_scheduler
import torch
        
def main():
    models = gen_models()
    scheduler = gen_scheduler()
    
    for model in tqdm(models):
        vae = AutoencoderKL.from_pretrained(model.path, subfolder="vae", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(model.path, subfolder="unet", torch_dtype=torch.float16)
        vae.save_pretrained(f'{model.name}/vae')
        unet.save_pretrained(f'{model.name}/unet')
        model.pipe_t2i = StableDiffusionPipeline.from_pretrained(model.path, unet=unet, vae=vae, torch_dtype=torch.float16, scheduler=scheduler)
        model.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(model.path, unet=unet, vae=vae, torch_dtype=torch.float16, scheduler=scheduler)
        model.pipe_t2i.save_pretrained(f'{model.name}/pipe_t2i')
        model.pipe_i2i.save_pretrained(f'{model.name}/pipe_i2i')
        
if __name__ == '__main__':
    main()