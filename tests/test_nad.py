import torch

from triggercraft.nad import AttentionDistiller, attention_map


def test_attention_transfer_with_named_layer():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2))
    images = torch.rand(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=2)
    distiller = AttentionDistiller(model, ["0"], beta=0.5)
    history = distiller.fit(loader, teacher_epochs=1, student_epochs=1)
    assert [record["stage"] for record in history] == ["teacher", "student"]
    assert attention_map(torch.rand(2, 4, 8, 8)).shape == (2, 8, 8)
