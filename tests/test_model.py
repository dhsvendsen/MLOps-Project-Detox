import pytest
import torch
import json
from src.models.model import LightningBertBinary
from tests import _PATH_DATA, _PROJECT_ROOT

n = 100
input = torch.load(f"{_PATH_DATA}/processed/tokens_test.pt")[:, 0:n, :].type(torch.int64)

with open(f"{_PROJECT_ROOT}/config/config.json") as file:
        cfg = json.load(file)
model = LightningBertBinary(cfg)
output = model(input[0], input[1])

@pytest.mark.parametrize("test_in,expected_out", [(10, 10), (20, 20), (42, 42)])
def test_output(test_in, expected_out):
    assert model(input[0,0:test_in], input[1,0:test_in]).shape[0] == expected_out

def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match="Expected tokens to a 2D tensor"):
        wrong_dim_inp = torch.randn(1, 2, 3, 2)
        model(wrong_dim_inp, wrong_dim_inp)
