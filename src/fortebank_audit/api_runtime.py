"""Compose the existing checked ONNX pipeline for the public API."""
from __future__ import annotations
import base64
import logging
import os
from pathlib import Path
import threading

ROOT=Path(__file__).resolve().parents[2]
_lock=threading.Lock()
_runtime=None
LOGGER=logging.getLogger(__name__)


class Runtime:
    def __init__(self):
        from .inference.benchmark import build_predictor,load_pipeline_config
        from .inference.semantic_pipeline import build_semantic_verifier_from_env
        _,self.config,_=load_pipeline_config()
        self.predictor=build_predictor('onnx',prefer_cuda=False)
        self.verifier=build_semantic_verifier_from_env()
        self.sequence=None
        adapter=os.environ.get('QAITU_SEQUENCE_ADAPTER')
        if adapter:
            if self.verifier is not None:
                raise ValueError('Choose semantic Qwen or sequence Qwen, not two GPU models')
            from .inference.qwen_sequence import SequenceClassifier
            self.sequence=SequenceClassifier(
                os.environ.get('QAITU_VLM_MODEL_PATH',str(ROOT/'models/pretrained/qwen3-vl-2b-instruct')),
                adapter,allow_experimental=os.environ.get('QAITU_ALLOW_EXPERIMENTAL_ADAPTER')=='1')
        self.evidence_root=Path(os.environ.get('FORTEBANK_EVIDENCE_ROOT',str(ROOT/'data/raw/evidence/api')))

    def analyze(self,video):
        from .inference.pipeline import analyze_video
        from .inference.semantic_pipeline import analyze_video_with_semantic_verification
        if self.verifier is None:
            result=analyze_video(video,self.predictor,self.config,evidence_root=self.evidence_root)
        else:
            result=analyze_video_with_semantic_verification(video,self.predictor,self.config,
                self.verifier,evidence_root=self.evidence_root)
        sequence=self.sequence.predict(video) if self.sequence else None
        if sequence:
            # Whole-video classification has no grounding/localization. Never use it
            # to silently remove detector events or invent timestamped events.
            LOGGER.info('Sequence review decision=%s gate=%s',sequence['decision'],sequence['quality_gate_passed'])
        violations=[]
        for event in result['events']:
            verification=event.get('semantic_verification')
            description=('Обнаружены признаки возможного нарушения. '
                         'Локальный детектор выделил фрагмент для проверки сотрудником.')
            if verification:
                description=f"{verification['thesis']}. {verification['reasoning']}. {verification['risk']}"
            if sequence:
                description+=(' Дополнительный экспериментальный анализ полного видео: '+
                    ('признаки возможного нарушения.' if sequence['decision']=='possible_violation'
                     else 'обычная активность; выводы моделей расходятся, требуется ручная проверка.'))
            evidence=[]
            for frame in event['evidence']:
                path=(self.evidence_root/frame['jpeg_path']).resolve()
                if not path.is_relative_to(self.evidence_root.resolve()):
                    raise ValueError('Invalid evidence path')
                evidence.append({'timestamp_sec':frame['timecode_sec'],
                    'image_base64':base64.b64encode(path.read_bytes()).decode('ascii')})
            if not 1<=len(evidence)<=3:
                raise ValueError('Evidence count outside API contract')
            violations.append({'type':'Признаки возможного нарушения',
                'timecode_sec':event['peak_time_sec'],'description':description,
                'evidence_frames':evidence})
        return {'verdict':'violation' if violations else 'no_violation','violations':violations}


def analyze(video_path,model_path=None):
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime=Runtime()
        return _runtime.analyze(video_path)
