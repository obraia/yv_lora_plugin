import gc
import os
import random
import torch
from safetensors.torch import load_file
from diffusers import pipelines, schedulers

torch.backends.cuda.matmul.allow_tf32 = True

class Lora:

  def __init__(self, models_path, lora_models_path, device):
    self.device = device
    self.models_path = models_path
    self.lora_models_path = lora_models_path

  def load_scheduler(self, sampler, config):
      match sampler:
          case 'euler_a':
              scheduler = schedulers.EulerAncestralDiscreteScheduler.from_config(config)
          case 'euler':
              scheduler = schedulers.EulerDiscreteScheduler.from_config(config)
          case 'ddim':
              scheduler = schedulers.DDIMScheduler.from_config(config)
          case 'ddpm':
              scheduler = schedulers.DDPMScheduler.from_config(config)
          case 'uni_pc':
              scheduler = schedulers.UniPCMultistepScheduler.from_config(config)
          case _:
              raise ValueError("Invalid sampler type")

      return scheduler
      
  def load_model(self, type, model):
      match type:
          case 'txt2img':
              pipe = pipelines.StableDiffusionPipeline.from_pretrained(
                  os.path.join(self.models_path, model),
                  revision="fp16",
                  safety_checker=None,
                  torch_dtype=torch.float16,
              )
          case _:
              raise ValueError("Invalid model type")
          
      return pipe

  def load_generator(self, seed):
      return torch.Generator(self.device).manual_seed(seed)

  def latents_callback(self, step, emit_progress):
      emit_progress({ "step": step })

  def random_seed(self):
      return random.randint(-9999999999, 9999999999)

  def txt2img(self, properties, lora_properties, emit_progress):
      pipe = self.load_model('txt2img', properties["model"])
      pipe.scheduler = self.load_scheduler(properties["sampler"], pipe.scheduler.config)

      for lora in lora_properties["loras"]:
          if lora["weight"] > 0:
              pipe = self.load_lora(pipe, lora["model"], lora["weight"])

      pipe.to(self.device)
      
      properties['seed'] = properties['seed'] if properties['seed'] != -1 else self.random_seed()
      generator = self.load_generator(properties["seed"])

      outputs = []

      for i in range(properties["images"]):
        output = pipe(
            prompt=properties["positive"],
            negative_prompt=properties["negative"],
            num_inference_steps=properties["steps"],
            guidance_scale=properties["cfg"],
            num_images_per_prompt=1,
            width=properties["width"],
            height=properties["height"],
            generator=generator,
            callback_steps=1,
            callback=lambda s, _, l: self.latents_callback(((s + 1) + (i * properties["steps"])), emit_progress),
            eta=0.0,
        ).images[0]

        outputs.append(output)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

      del pipe
      del generator
      gc.collect()

      return outputs

  def load_lora(self, pipeline, model, lora_weight=0.5):
      state_dict = load_file(os.path.join(self.lora_models_path, model))
      LORA_PREFIX_UNET = 'lora_unet'
      LORA_PREFIX_TEXT_ENCODER = 'lora_te'

      alpha = lora_weight
      visited = []

      # directly update weight in diffusers model
      for key in state_dict:
          
          # as we have set the alpha beforehand, so just skip
          if '.alpha' in key or key in visited:
              continue
              
          if 'text' in key:
              layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
              curr_layer = pipeline.text_encoder
          else:
              layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
              curr_layer = pipeline.unet

          # find the target layer
          temp_name = layer_infos.pop(0)
          while len(layer_infos) > -1:
              try:
                  curr_layer = curr_layer.__getattr__(temp_name)
                  if len(layer_infos) > 0:
                      temp_name = layer_infos.pop(0)
                  elif len(layer_infos) == 0:
                      break
              except Exception:
                  if len(temp_name) > 0:
                      temp_name += '_'+layer_infos.pop(0)
                  else:
                      temp_name = layer_infos.pop(0)
          
          # org_forward(x) + lora_up(lora_down(x)) * multiplier
          pair_keys = []
          if 'lora_down' in key:
              pair_keys.append(key.replace('lora_down', 'lora_up'))
              pair_keys.append(key)
          else:
              pair_keys.append(key)
              pair_keys.append(key.replace('lora_up', 'lora_down'))
          
          # update weight
          if len(state_dict[pair_keys[0]].shape) == 4:
              weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
              weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
              curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
          else:
              weight_up = state_dict[pair_keys[0]].to(torch.float32)
              weight_down = state_dict[pair_keys[1]].to(torch.float32)
              curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
              
          # update visited list
          for item in pair_keys:
              visited.append(item)
          
      return pipeline