import pytest
import torch
from torch import nn

from timeseries.models.gnn.model import GNN, my_gcnn


def test_graph_layers_compose_and_preserve_adjacency():
    adjacency = torch.tril(torch.ones(5, 5), diagonal=-1)
    features = torch.randn(2, 5, 3, requires_grad=True)
    layers = nn.Sequential(my_gcnn(3, 4), my_gcnn(4, 2, relu=False))

    output, returned_adjacency = layers((features, adjacency))

    assert output.shape == (2, 5, 2)
    assert returned_adjacency is adjacency
    output.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def make_gnn(depth: int) -> GNN:
    return GNN(
        past_steps=200, future_steps=65, future_channel=0,
        embs=[4], cat_emb_dim=2, hidden_GNN=4, in_feat=4,
        past=200, future=65, emb={"category": (4, 2)}, device="cpu",
        nfeat_out_gnn=4, num_layer1=depth, num_layer2=depth,
        hid_out_features1=4, hid_out_features2=4,
    )


@pytest.mark.parametrize("depth", [1, 3])
def test_gnn_outputs_and_checkpoint_round_trip(depth):
    model = make_gnn(depth).eval()
    data = torch.randn(2, 265, 4)
    data[:, :, 0] = torch.randint(0, 4, (2, 265))

    with torch.no_grad():
        prediction, distances, adjacency = model(data)

    assert prediction.shape == (2, 65)
    assert distances.shape == (2, 265, 265)
    assert adjacency.shape == (265, 265)
    for value in (prediction, distances, adjacency):
        assert torch.isfinite(value).all()
    assert torch.count_nonzero(torch.triu(adjacency)) == 0

    restored = make_gnn(depth).eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        for actual, expected in zip(restored(data), (prediction, distances, adjacency), strict=True):
            torch.testing.assert_close(actual, expected)
