"""Separate experimental ComfyUI nodes; no production node modifications."""
import math
import os
import re
import json
import time
import uuid
from pathlib import Path
import torch
import nodes


class ExperimentalTailDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'samples': ('LATENT',), 'vae': ('VAE',),
            'tail_percent': ('FLOAT', {'default': 5.0, 'min': 1.0, 'max': 100.0, 'step': 1.0}),
            'minimum_latent_frames': ('INT', {'default': 4, 'min': 2, 'max': 256}),
            'tile_size': ('INT', {'default': 256, 'min': 64, 'max': 4096, 'step': 32}),
            'overlap': ('INT', {'default': 64, 'min': 0, 'max': 1024, 'step': 32}),
            'temporal_size': ('INT', {'default': 16, 'min': 8, 'max': 4096, 'step': 4}),
            'temporal_overlap': ('INT', {'default': 4, 'min': 4, 'max': 1024, 'step': 4})}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'STRING')
    RETURN_NAMES = ('원본 마지막 프레임', '디코딩한 끝부분', '실험 정보')
    FUNCTION = 'decode'
    CATEGORY = 'Video Studio/Experiments'

    def decode(self, samples, vae, tail_percent, minimum_latent_frames, tile_size, overlap, temporal_size, temporal_overlap):
        latent = samples['samples']
        if latent.is_nested:
            latent = latent.unbind()[0]
        if latent.ndim != 5 or latent.shape[0] != 1:
            raise ValueError('단일 영상의 5차원 잠재 데이터가 필요합니다.')
        total = int(latent.shape[2])
        if total < 2:
            raise ValueError('시간축 잠재 프레임이 최소 2개 필요합니다.')
        count = min(total, max(minimum_latent_frames, math.ceil(total * tail_percent / 100)))
        cut = dict(samples, samples=latent[:, :, total-count:].contiguous())
        frames = nodes.VAEDecodeTiled().decode(vae, cut, tile_size, overlap, temporal_size, temporal_overlap)[0]
        if len(frames) == 0:
            raise RuntimeError('끝부분 디코딩 결과가 비어 있습니다.')
        return (frames[-1:].clone(), frames, f'latent {count}/{total}; 독립 부분 디코딩은 전체 결과와 다를 수 있습니다.')


class ExperimentalTailCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'early_frame': ('IMAGE',), 'full_video': ('IMAGE',)}}
    RETURN_TYPES = ('STRING', 'IMAGE')
    RETURN_NAMES = ('비교 결과', '차이 영상')
    FUNCTION = 'compare'
    CATEGORY = 'Video Studio/Experiments'
    def compare(self, early_frame, full_video):
        a, b = early_frame[-1:], full_video[-1:]
        if a.shape != b.shape:
            raise ValueError('프레임 크기가 다릅니다.')
        difference = (a.float() - b.float()).abs()
        return (f'MAE={difference.mean().item():.6f}; max={difference.max().item():.6f}; 0에 가까울수록 일치', (difference * 4).clamp(0, 1))


class ExperimentalBoundaryBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'frames': ('IMAGE',), 'anchor': ('IMAGE',),
                             'blend_frames': ('INT', {'default': 4, 'min': 1, 'max': 120})}}
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('시작 경계 보정 영상',)
    FUNCTION = 'blend'
    CATEGORY = 'Video Studio/Experiments'
    def blend(self, frames, anchor, blend_frames):
        if frames.ndim != 4 or anchor.ndim != 4 or not len(frames) or not len(anchor):
            raise ValueError('이미지 프레임이 필요합니다.')
        if frames.shape[1:] != anchor.shape[1:]:
            raise ValueError('영상과 기준 프레임 크기가 일치해야 합니다.')
        count = min(blend_frames, len(frames))
        result = frames.clone()
        weights = torch.linspace(1.0, 0.0, count + 1, device=frames.device, dtype=frames.dtype)[:-1].reshape(-1, 1, 1, 1)
        result[:count] = weights * anchor[-1:].to(frames.device, frames.dtype) + (1 - weights) * frames[:count]
        return (result,)


class ExperimentalSamplerProfile:
    """Profile one entire sampling call. Trace support varies by XPU runtime."""
    @classmethod
    def INPUT_TYPES(cls):
        base=nodes.NODE_CLASS_MAPPINGS['StudioSamplerCustomAdvanced'].INPUT_TYPES()
        return {'required': dict(base['required']), 'optional': dict(base.get('optional', {}))}
    RETURN_TYPES = ('LATENT', 'LATENT', 'STRING')
    RETURN_NAMES = ('샘플', '노이즈 제거 샘플', '계측 기록 경로')
    FUNCTION = 'profile'
    CATEGORY = 'Video Studio/Experiments'
    def profile(self, device, spill_to_peer=True, **kwargs):
        import folder_paths
        activities=[torch.profiler.ProfilerActivity.CPU]
        supported=torch.profiler.supported_activities()
        xpu=getattr(torch.profiler.ProfilerActivity, 'XPU', None)
        if xpu is not None and xpu in supported:
            activities.append(xpu)
        target=Path(folder_paths.get_output_directory())/'video-studio'/'experimental-profiles'
        target.mkdir(parents=True,exist_ok=True)
        path=target/('sampling-'+uuid.uuid4().hex+'.json')
        if str(device).startswith('xpu'):
            torch.xpu.synchronize(torch.device(device))
        start=time.monotonic()
        with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=True, with_stack=False) as profiler:
            result=nodes.NODE_CLASS_MAPPINGS['StudioSamplerCustomAdvanced']().sample(device=device, spill_to_peer=spill_to_peer, **kwargs)
            if str(device).startswith('xpu'):
                torch.xpu.synchronize(torch.device(device))
        profiler.export_chrome_trace(str(path))
        detail=f'{path} | {time.monotonic()-start:.2f}s | XPU profiler: {xpu in activities}'
        return (result[0],result[1],detail)


class ExperimentalSaveAVLatent:
    """Persist only AV samples for local handoff; no pickle or prompt metadata."""
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'samples': ('LATENT',)}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('잠재 데이터 ID',)
    FUNCTION = 'save'
    OUTPUT_NODE = True
    CATEGORY = 'Video Studio/Experiments'
    def save(self, samples):
        import folder_paths
        from safetensors.torch import save_file
        value=samples['samples']
        if not value.is_nested:
            raise ValueError('영상과 오디오가 함께 있는 MiniMax 잠재 데이터가 필요합니다.')
        parts=value.unbind()
        if len(parts)!=2 or parts[0].ndim!=5 or parts[1].ndim!=4:
            raise ValueError('지원하지 않는 영상·오디오 잠재 데이터 형식입니다.')
        directory=Path(folder_paths.get_output_directory())/'video-studio'/'experimental-latents'
        directory.mkdir(parents=True,exist_ok=True)
        token=uuid.uuid4().hex
        target=directory/(token+'.safetensors')
        temporary=directory/(token+'.tmp')
        try:
            save_file({'video':parts[0].detach().to('cpu').contiguous(),
                       'audio':parts[1].detach().to('cpu').contiguous()},str(temporary))
            os.replace(temporary,target)
        finally:
            if temporary.exists():temporary.unlink()
        return {'ui': {'text':[token]}, 'result':(token,)}


class ExperimentalLoadAVLatent:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'token': ('STRING', {'default':''})}}
    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('영상·오디오 잠재 데이터',)
    FUNCTION = 'load'
    CATEGORY = 'Video Studio/Experiments'
    def load(self, token):
        import folder_paths
        import comfy.nested_tensor
        from safetensors.torch import load_file
        if not re.fullmatch('[0-9a-f]{32}', token):raise ValueError('잠재 데이터 ID 형식이 올바르지 않습니다.')
        root=(Path(folder_paths.get_output_directory())/'video-studio'/'experimental-latents').resolve()
        path=(root/(token+'.safetensors')).resolve()
        if path.parent!=root:raise ValueError('잘못된 파일 경로입니다.')
        values=load_file(str(path),device='cpu')
        if set(values)!={'video','audio'} or values['video'].ndim!=5 or values['audio'].ndim!=4:
            raise ValueError('지원하지 않는 영상·오디오 잠재 파일입니다.')
        return ({'samples':comfy.nested_tensor.NestedTensor((values['video'],values['audio']))},)


class ExperimentalSaveConditioning:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'conditioning':('CONDITIONING',)}}
    RETURN_TYPES=('STRING',)
    RETURN_NAMES=('텍스트 인코딩 ID',)
    FUNCTION='save'
    OUTPUT_NODE=True
    CATEGORY='Video Studio/Experiments'
    def save(self,conditioning):
        import folder_paths
        from safetensors.torch import save_file
        tensors={}
        def pack(value):
            if isinstance(value,torch.Tensor):
                key=f'tensor_{len(tensors)}';tensors[key]=value.detach().to('cpu').contiguous()
                return {'type':'tensor','key':key}
            if isinstance(value,dict):
                if not all(isinstance(k,str) for k in value):raise ValueError('Unsupported conditioning key')
                return {'type':'dict','items':{k:pack(v) for k,v in value.items()}}
            if isinstance(value,(list,tuple)):
                return {'type':'tuple' if isinstance(value,tuple) else 'list','items':[pack(v) for v in value]}
            if value is None or isinstance(value,(str,int,float,bool)):
                return {'type':'value','value':value}
            raise ValueError(f'Unsupported conditioning value: {type(value).__name__}')
        tree=pack(conditioning)
        if not tensors:raise ValueError('Conditioning contains no tensors')
        directory=Path(folder_paths.get_output_directory())/'video-studio'/'experimental-conditioning'
        directory.mkdir(parents=True,exist_ok=True)
        token=uuid.uuid4().hex; target=directory/(token+'.safetensors');temporary=directory/(token+'.tmp')
        try:
            save_file(tensors,str(temporary),metadata={'tree':json.dumps(tree,ensure_ascii=False,allow_nan=False)})
            os.replace(temporary,target)
        finally:
            if temporary.exists():temporary.unlink()
        return {'ui':{'text':[token]},'result':(token,)}


class ExperimentalLoadConditioning:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'token':('STRING',{'default':''})}}
    RETURN_TYPES=('CONDITIONING',)
    RETURN_NAMES=('미리 인코딩한 텍스트',)
    FUNCTION='load'
    CATEGORY='Video Studio/Experiments'
    def load(self,token):
        import folder_paths
        from safetensors import safe_open
        if not re.fullmatch('[0-9a-f]{32}',token):raise ValueError('Invalid conditioning ID')
        root=(Path(folder_paths.get_output_directory())/'video-studio'/'experimental-conditioning').resolve()
        path=(root/(token+'.safetensors')).resolve()
        if path.parent!=root:raise ValueError('Invalid conditioning path')
        with safe_open(str(path),framework='pt',device='cpu') as f:
            tree=json.loads(f.metadata()['tree'])
            def unpack(node):
                kind=node['type']
                if kind=='tensor':return f.get_tensor(node['key'])
                if kind=='dict':return {k:unpack(v) for k,v in node['items'].items()}
                if kind in ('list','tuple'):
                    values=[unpack(v) for v in node['items']]
                    return tuple(values) if kind=='tuple' else values
                if kind=='value':return node['value']
                raise ValueError('Unsupported conditioning item')
            result=unpack(tree)
        if not isinstance(result,list):raise ValueError('Invalid conditioning format')
        return (result,)


NODE_CLASS_MAPPINGS = {'ExperimentalTailDecode': ExperimentalTailDecode, 'ExperimentalTailCompare': ExperimentalTailCompare, 'ExperimentalBoundaryBlend': ExperimentalBoundaryBlend, 'ExperimentalSamplerProfile': ExperimentalSamplerProfile, 'ExperimentalSaveAVLatent': ExperimentalSaveAVLatent, 'ExperimentalLoadAVLatent': ExperimentalLoadAVLatent, 'ExperimentalSaveConditioning': ExperimentalSaveConditioning, 'ExperimentalLoadConditioning': ExperimentalLoadConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {'ExperimentalTailDecode': '실험 · 마지막 프레임 먼저 디코딩', 'ExperimentalTailCompare': '실험 · 마지막 프레임 차이 비교', 'ExperimentalBoundaryBlend': '실험 · 클립 시작 경계 블렌딩', 'ExperimentalSamplerProfile': '실험 · 샘플링 상세 계측', 'ExperimentalSaveAVLatent':'실험 · 영상 오디오 잠재 저장', 'ExperimentalLoadAVLatent':'실험 · 영상 오디오 잠재 불러오기', 'ExperimentalSaveConditioning':'실험 · 텍스트 인코딩 저장', 'ExperimentalLoadConditioning':'실험 · 텍스트 인코딩 불러오기'}
