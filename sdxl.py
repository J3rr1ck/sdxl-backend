import torch
from diffusers import DiffusionPipeline
import platform

def torch_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"

def pipeline(
    model="stabilityai/stable-diffusion-xl-base-1.0",
    device=torch_device(),
    watermark=False,
    low_vram=False,
):
    torch_dtype = torch.float16
    variant = "fp16"

    # MacOS can only use fp32
    if device == "mps":
        torch_dtype = torch.float32
        variant = "fp32"
    pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant=variant,
    )

    # enable VAE titling and slicing if low VRAM
    if low_vram:
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()

    # model offloading to save memory
    if low_vram and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # Apply torch.compile only if OS is not Windows
    if platform.system() != "Windows":
        pipe.unit = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    # mock out watermark if needed
    if not watermark:
        pipe.watermark = NoWatermark()

    return pipe

class NoWatermark:
    def apply_watermark(self, img):
        return img
