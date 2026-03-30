from diffusers import WanPipeline, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import torch, os, gc
import torch.nn.functional as F
import argparse
from modify_model.modify_wan import SageWanAttnProcessor,set_sage_attn_wan
from tqdm import tqdm
from sageattention import sageattn
from contextlib import nullcontext
from flash_attn import flash_attn_func, flash_attn_varlen_func
ATTENTION = {
    "sage": sageattn,
    "sdpa": F.scaled_dot_product_attention,
    "fa": flash_attn_func
}

from xfuser.core.distributed import (
    get_world_group,
    initialize_model_parallel,
    init_distributed_environment,
)

from xfuser.core.distributed.parallel_state import (
    model_parallel_is_initialized,
    destroy_distributed_environment,
    destroy_model_parallel,
)
from PIL import Image
import torch.distributed as dist
import logging
from xmagic.process import add_argument, profile_pipeline_modules, profile_wrapper, init_logging

# os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ["TOKENIZERS_PARALLELISM"]="false"
negative_prompt_1 = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
negative_prompt_2 = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
prompt_path = "videos/testing_prompts.txt"

from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.transformers.transformer_wan import WanTransformerBlock, WanTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
try:
    from xfuser.core.distributed import (
        get_ulysses_parallel_world_size,
        get_ulysses_parallel_rank,
        get_sp_group
    )
except:
    pass


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int, value: float = 0.0):
    """
    Pad tensor x on dimension dim so that size(dim) is a multiple of `multiple`
    """
    size = x.size(dim)
    pad_len = (multiple - size % multiple) % multiple
    if pad_len == 0:
        return x, 0

    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad_tensor = x.new_full(pad_shape, value)

    x = torch.cat([x, pad_tensor], dim=dim)
    return x, pad_len

class Wan22Transformer3DModel_Sparse(WanTransformer3DModel):
    """
    Wan2.2 compatible 3D transformer model with sparse attention support.
    Handles expand_timesteps feature for fine-grained temporal control.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logging.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v with expand_timesteps)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        pad_len = 0
        if dist.is_initialized() and get_ulysses_parallel_world_size() > 1:
            hidden_states, pad_len = pad_to_multiple(hidden_states, get_ulysses_parallel_world_size(), dim=1, value=0.0)
            rotary_emb_0, pad_len = pad_to_multiple(rotary_emb[0], get_ulysses_parallel_world_size(), dim=1, value=0.0) 
            rotary_emb_1, pad_len = pad_to_multiple(rotary_emb[1], get_ulysses_parallel_world_size(), dim=1, value=0.0) 
            if pad_len > 0:
                SageWanAttnProcessor.pad_len = pad_len
            # split video latents on dim TS
            hidden_states = torch.chunk(hidden_states, get_ulysses_parallel_world_size(), dim=-2)[get_ulysses_parallel_rank()]
            rotary_emb = (
                torch.chunk(rotary_emb_0, get_ulysses_parallel_world_size(), dim=1)[get_ulysses_parallel_rank()],
                torch.chunk(rotary_emb_1, get_ulysses_parallel_world_size(), dim=1)[get_ulysses_parallel_rank()],
            )

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states, 
                    encoder_hidden_states, 
                    timestep_proj, 
                    rotary_emb,
                )

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v with expand_timesteps)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            
        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if dist.is_initialized() and get_ulysses_parallel_world_size() > 1:
            hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)

        if pad_len > 0:
            hidden_states = hidden_states[:, :-pad_len, ...].contiguous()

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

@profile_wrapper
def i2v_eval_infer(warm_round, eval_round, prompt, image, pipe, generator, height, width, args):
    """
    执行推理评估函数。

    Args:
        warm_round (int): 预热轮次。
        eval_round (int): 评估轮次。
        pipe (DiffusersPipeline): 推理管道。
        generator (Union[VQGanModel, StableDiffusionPipeline]): 生成器模型。
        args (Namespace): 命令行参数。

    Returns:
        PIL.Image.Image: 推理生成的视频帧图像。

    """
    video = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt_2,
        height=height,
        width=width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).frames[0]
    return video

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model", choices=["wan2.1-14b", "wan2.2-14b"], default="wan2.1-1.3b", help="Wan model")
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument('--attention_type', type=str, default='sdpa', choices=['sdpa', 'sage', 'fa'], help='Attention type')
    parser.add_argument("--start", type=int, default=0, help="Starting prompt id of this run.")
    parser.add_argument("--end", type=int, default=12, help="Ending prompt id of this run.")
    
    parser.add_argument("--warmup_round", type=int, default=1, help="warmup round")
    parser.add_argument("--eval_round", type=int, default=10, help="eval round")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random generator to get consistent results")
   
    parser.add_argument(
        "--use_cfg_parallel",
        action="store_true",
        help="Use split batch in classifier_free_guidance. cfg_degree will be 2 if set",
    )
    parser.add_argument("--use_sequence_parallel", action="store_true",default=False, help="Enable sequence parallelism for parallel inference")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Ulysses size")
    parser.add_argument("--profile", action="store_true", help="profile modules")
    parser.add_argument("--num_frames", type=int, default=81, help="num frames for video")
    parser.add_argument("--fps", type=int, default=16, help="fps for video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="num_images_per_prompt per prompt")
    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video") 
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--guidance_scale_2", type=float, default=3.0, help="Guidance scale for the second transformer (Wan2.2)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    if world_size > 1:
        # init distribute env
        init_distributed_environment(backend="nccl", distributed_init_method="env://")
        rank = get_world_group().rank
        world_size = get_world_group().world_size
        local_rank = get_world_group().local_rank

    log_level = logging.DEBUG if args.profile else logging.INFO
    init_logging(rank, log_level)
    
    logging.info("==== Parsed Args ====")
    for k, v in sorted(vars(args).items()):
        logging.info(f"{k:<20} = {v}")
    logging.info("=====================")
    
    use_cfg_parallel = args.use_cfg_parallel
    cfg_degree = 2 if use_cfg_parallel else 1
    ulysses_degree = args.ulysses_size
    use_sp = False
    if world_size > 1:
        degree = cfg_degree * ulysses_degree
        if degree > world_size:
            logging.error("world_size must greater than ulysses_size = {}".format(args.ulysses_size))
        use_sp = True
        initialize_model_parallel(
            data_parallel_degree=1,
            classifier_free_guidance_degree=cfg_degree,
            ulysses_degree=ulysses_degree,
            sequence_parallel_degree=ulysses_degree,
        )
    else:
        logging.info("single gpu mode")


    if args.model == "wan2.1-14b": model_path = "/workspace/hub/wan/Wan2.1-I2V-14B-Diffusers"
    else: model_path = "/workspace/hub/wan/Wan2.2-I2V-A14B-Diffusers"

    model_path = "/workspace/hub/wan/Wan2.2-I2V-A14B-Lightning-Diffusers/"
    if rank == 0:
        video_dir = f"videos/{args.model}/{args.attention_type}"
        os.makedirs(video_dir, exist_ok=True)

    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

    # with open(prompt_path, "r", encoding="utf-8") as file:
    #     prompts = file.readlines()
    # selected_prompts = [p.strip() for p in prompts[args.start:args.end]]

    pipe = WanImageToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    set_sage_attn_wan(pipe.transformer, ATTENTION[args.attention_type])
    if getattr(pipe, "transformer_2", None) is not None: # Wan2.2
        set_sage_attn_wan(pipe.transformer_2, ATTENTION[args.attention_type])
    WanTransformer3DModel.forward = Wan22Transformer3DModel_Sparse.forward
 
    pipe.to("cuda")

     # device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device='cpu').manual_seed(42)

    # if args.compile:
    #     pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
    #     if getattr(pipe, "transformer_2", None) is not None: # Wan2.2
    #         pipe.transformer_2 = torch.compile(pipe.transformer_2, mode="max-autotune-no-cudagraphs")
    if args.profile:
        modules = ["text_encoder", "text_encoder_2", "unet", "vae", "transformer", "transformer_2"]
        pipe = profile_pipeline_modules(pipe=pipe, modules=modules, rank=rank)

    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_tiling()
    # pipe.vae.enable_slicing()

    import numpy as np
    width = args.width
    height = args.height
    max_area = width * height
    img = load_image("resource/wan_images/wan_i2v_input.JPG")
    #aspect_ratio = img.height / img.width
    #mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    #height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    #width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    #logging.info(f"Image's width*height after mod: {width}*{height}")
    #img = img.resize((width, height))
    ori_width, ori_height = img.size
    width_ratio = width / ori_width
    height_ratio = height / ori_height
    scale_ratio = min(width_ratio, height_ratio)
    new_width = int(ori_width * scale_ratio)
    new_height = int(ori_height * scale_ratio)

    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_canvas = Image.new("RGB", (width, height), (0, 0, 0))

    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2
    img_canvas.paste(img_resized, (paste_x, paste_y))
    
    video = i2v_eval_infer(args.warmup_round, args.eval_round, prompt, img_canvas, pipe, gen, height, width, args)

    logging.info("Run over!")

    # save video
    video_name = (
        "wan_2.2_14b_i2v-"
        + str(args.num_frames)
        + "x"
        + str(width)
        + "x"
        + str(height)
        + "-steps"
        + str(args.num_inference_steps)
        + "-sp"
        + str(args.ulysses_size)
        + "-cfg"
        + str(cfg_degree)
        + str(f"-{args.attention_type}")
        + ".mp4"
    )
    export_to_video(video, video_name, fps=16)

    if model_parallel_is_initialized():
        destroy_model_parallel()
    destroy_distributed_environment()
