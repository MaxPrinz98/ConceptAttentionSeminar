
import torch

try:
    if not torch.backends.mps.is_available():
        print("MPS not available")
        exit(0)

    device = torch.device("mps")
    print(f"Testing on {device}")

    # Test randn in bfloat16
    try:
        noise = torch.randn(1, 4, 64, 64, device=device, dtype=torch.bfloat16)
        print(f"randn bfloat16: Success, mean={noise.mean().item()}, std={noise.std().item()}")
    except Exception as e:
        print(f"randn bfloat16: Failed - {e}")

    # Test Conv2d in bfloat16
    try:
        conv = torch.nn.Conv2d(4, 4, 3, padding=1).to(device).to(torch.bfloat16)
        out = conv(noise)
        print(f"Conv2d bfloat16: Success, out mean={out.mean().item()}")
    except Exception as e:
        print(f"Conv2d bfloat16: Failed - {e}")

    # Test mixed precision Conv2d (input fp32, weight bf16)
    try:
        conv = torch.nn.Conv2d(4, 4, 3, padding=1).to(device).to(torch.bfloat16)
        inp = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
        out = conv(inp)
        print(f"Conv2d mixed (in fp32, w bf16): Success, out mean={out.mean().item()}, dtype={out.dtype}")
    except Exception as e:
        print(f"Conv2d mixed: Failed - {e}")

except Exception as e:
    print(f"General Error: {e}")
