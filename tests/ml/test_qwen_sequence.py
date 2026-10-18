import importlib.util
from pathlib import Path
from fortebank_audit.inference.qwen_sequence import class_token_ids,FRAME_COUNT


def test_confusion_counts_include_false_negatives():
    path=Path(__file__).resolve().parents[2]/'scripts/ml/train_qwen_sequence.py'
    spec=importlib.util.spec_from_file_location('sequence_training',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result=module.metrics([{'label':y,'prediction':p} for y,p in [(1,1),(1,0),(0,1),(0,0)]])
    assert result=={'tp':1,'tn':1,'fp':1,'fn':1,'macro_f1':.5,'recall':.5,'count':4}


def test_class_token_contract():
    import pytest
    class Tokenizer:
        def encode(self,value,add_special_tokens):
            return [32] if value=='A' else [33]
    class Processor:
        tokenizer=Tokenizer()
    assert class_token_ids(Processor())==[32,33]
    Processor.tokenizer.encode=lambda *args,**kwargs:[1,2]
    with pytest.raises(ValueError,match='single tokens'):
        class_token_ids(Processor())
    assert FRAME_COUNT==16
