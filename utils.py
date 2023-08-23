from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import numpy as np
    
class Model:
    def __init__(self, name, path="", prefix=""):
        self.name = name
        self.path = path
        self.prefix = prefix
        self.pipe_t2i = None
        self.pipe_i2i = None

def load_model(model, scheduler):
    vae = AutoencoderKL.from_pretrained(f'{model.name}/vae', subfolder="vae", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(f'{model.name}/unet', subfolder="unet", torch_dtype=torch.float16)
    model.pipe_t2i = StableDiffusionPipeline.from_pretrained(f'{model.name}/pipe_t2i', unet=unet, vae=vae, torch_dtype=torch.float16, scheduler=scheduler)
    model.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(f'{model.name}/pipe_i2i', unet=unet, vae=vae, torch_dtype=torch.float16, scheduler=scheduler)
    return model

def inference(model, last_mode, prompt, guidance, steps, device, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt=""):
    generator = torch.Generator(device).manual_seed(seed) if seed != 0 else None
    assert last_mode in ['txt2img', 'img2img'], 'last mode should be in [\'txt2img\', \'img2img\']'
    if last_mode == 'txt2img':
        return txt_to_img(model, prompt, neg_prompt, guidance, steps, width, height, generator, device)
    else:
        return img_to_img(model, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator, device)
    
def txt_to_img(model, prompt, neg_prompt, guidance, steps, width, height, generator, device):
    prompt = model.prefix + prompt
    model.pipe_t2i = model.pipe_t2i.to(device)
    result = model.pipe_t2i(
      prompt,
      negative_prompt = neg_prompt,
      # num_images_per_prompt=n_images,F
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator) 
    return result

def img_to_img(model, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator, device):
    prompt = model.prefix + prompt
    img = Image.open(img)
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    model.pipe_i2i = model.pipe_i2i.to(device)
    result = model.pipe_i2i(
        prompt,
        negative_prompt = neg_prompt,
        # num_images_per_prompt=n_images,
        image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        generator = generator)
    return result

def gen_models():
    models = [
    Model("anything v3", "Linaqruf/anything-v3.0", "anything v3 style"),
    Model("Spider-Verse", "nitrosocke/spider-verse-diffusion", "spiderverse style "),
    Model("Balloon Art", "Fictiverse/Stable_Diffusion_BalloonArt_Model", "BalloonArt "),
    Model("Elden Ring", "nitrosocke/elden-ring-diffusion", "elden ring style "),
    Model("Tron Legacy", "dallinmackay/Tron-Legacy-diffusion", "trnlgcy "),
    Model("Pokémon", "lambdalabs/sd-pokemon-diffusers", ""),
    Model("Pony Diffusion", "AstraliteHeart/pony-diffusion", ""),
    Model("Robo Diffusion", "nousr/robo-diffusion", "")
    ]
    return models

def gen_scheduler():
    scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    trained_betas=None,
    predict_epsilon=True,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
    )
    return scheduler
