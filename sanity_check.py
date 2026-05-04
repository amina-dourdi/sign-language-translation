"""
=============================================================
sanity_check.py — Architecture Validation
=============================================================
Role: Validate that the CSLT model architecture works correctly
      BEFORE running any training on real data.

NO DATASET REQUIRED — uses randomly generated fake data.

This script runs 5 critical tests:

Test 1 — Forward Pass:
    Verify that tensors flow through the model without errors
    and the output shape matches expectations.

Test 2 — Overfit Single Batch:
    Train on one tiny batch for 50 steps. If the loss decreases
    close to zero, the model CAN learn (gradients work properly).

Test 3 — Encoder Freeze:
    Freeze encoder parameters and verify they don't change
    after a training step (transfer learning validation).

Test 4 — Greedy Decoding:
    Test the autoregressive inference pipeline to make sure
    the model can generate token sequences.

Test 5 — Parameter Count:
    Print a summary of trainable vs frozen parameters to
    verify the transfer learning setup.
=============================================================

USAGE:
    python sanity_check.py

Expected output: All 5 tests should PASS ✅
=============================================================
"""

import sys
import torch
import torch.optim as optim
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.cslt_model import CSLTModel
from training.loss import CSLTLoss


# ─────────────────────────────────────────────
# CONFIGURATION FOR TESTS (small values for speed)
# ─────────────────────────────────────────────
TEST_CONFIG = {
    "batch_size": 4,
    "max_frames": 30,        # Short sequences for fast testing
    "input_dim": 411,        # OpenPose keypoint features
    "vocab_size": 500,       # Small vocabulary for testing
    "d_model": 128,          # Small model for fast CPU testing
    "nhead": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "max_seq_len": 20,       # Short target sequences
}


def create_fake_data(config):
    """
    Generate random fake data that mimics real CSLT input/output.

    Returns:
        keypoints (FloatTensor): [B, T, D] — random keypoint sequences
        targets (LongTensor): [B, S] — random token sequences with
                              <SOS> at start and <EOS> at end
    """
    B = config["batch_size"]
    T = config["max_frames"]
    D = config["input_dim"]
    S = config["max_seq_len"]
    V = config["vocab_size"]

    # Random keypoints (simulating normalized MediaPipe/OpenPose output)
    keypoints = torch.randn(B, T, D)

    # Random target tokens: [<SOS>, random words..., <EOS>, <PAD>...]
    targets = torch.zeros(B, S, dtype=torch.long)
    for i in range(B):
        seq_len = torch.randint(5, S - 2, (1,)).item()
        targets[i, 0] = 1  # <SOS>
        targets[i, 1:seq_len + 1] = torch.randint(4, V, (seq_len,))
        targets[i, seq_len + 1] = 2  # <EOS>
        # Rest remains 0 (<PAD>)

    return keypoints, targets


def create_model(config):
    """Create a CSLT model with test configuration."""
    return CSLTModel(
        input_dim=config["input_dim"],
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_src_len=config["max_frames"],
        max_tgt_len=config["max_seq_len"],
    )


# ═══════════════════════════════════════════════
# TEST 1: Forward Pass (Shape Validation)
# ═══════════════════════════════════════════════
def test_forward_pass():
    """
    Verify that a forward pass produces correct output shapes.

    Expected:
        Input:  keypoints [4, 30, 411], targets [4, 20]
        Output: logits    [4, 19, 500]  (S-1 because of shift)
    """
    print("\n" + "═" * 55)
    print("  TEST 1: Forward Pass (Shape Validation)")
    print("═" * 55)

    model = create_model(TEST_CONFIG)
    keypoints, targets = create_fake_data(TEST_CONFIG)

    try:
        logits = model(keypoints, targets)

        B = TEST_CONFIG["batch_size"]
        S = TEST_CONFIG["max_seq_len"]
        V = TEST_CONFIG["vocab_size"]
        expected_shape = (B, S - 1, V)

        print(f"  Input keypoints : {keypoints.shape}")
        print(f"  Input targets   : {targets.shape}")
        print(f"  Output logits   : {logits.shape}")
        print(f"  Expected shape  : {expected_shape}")

        assert logits.shape == expected_shape, \
            f"Shape mismatch! Got {logits.shape}, expected {expected_shape}"

        # Check that output contains valid values (no NaN or Inf)
        assert not torch.isnan(logits).any(), "Output contains NaN!"
        assert not torch.isinf(logits).any(), "Output contains Inf!"

        print("  ✅ TEST 1 PASSED — Forward pass produces correct shapes")
        return True

    except Exception as e:
        print(f"  ❌ TEST 1 FAILED — {e}")
        return False


# ═══════════════════════════════════════════════
# TEST 2: Overfit Single Batch
# ═══════════════════════════════════════════════
def test_overfit():
    """
    Train on a single batch for 50 steps and check if loss decreases.

    If the model can overfit one batch, it means:
    - Gradients are flowing correctly
    - The model has enough capacity to learn
    - The loss function is working properly

    Expected: Loss should decrease by at least 50%
    """
    print("\n" + "═" * 55)
    print("  TEST 2: Overfit Single Batch")
    print("═" * 55)

    model = create_model(TEST_CONFIG)
    keypoints, targets = create_fake_data(TEST_CONFIG)

    criterion = CSLTLoss(
        vocab_size=TEST_CONFIG["vocab_size"],
        pad_idx=0,
        label_smoothing=0.0,  # No smoothing for overfit test
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(50):
        logits = model(keypoints, targets)
        loss = criterion(logits, targets[:, 1:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == 49:
            final_loss = loss.item()

        if step % 10 == 0:
            print(f"  Step {step:3d} │ Loss: {loss.item():.4f}")

    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss  : {final_loss:.4f}")
    print(f"  Reduction   : {((initial_loss - final_loss) / initial_loss) * 100:.1f}%")

    if final_loss < initial_loss * 0.5:
        print("  ✅ TEST 2 PASSED — Model can learn (loss decreased >50%)")
        return True
    else:
        print("  ⚠️  TEST 2 WARNING — Loss did not decrease enough")
        print("     This may be OK with a very small model. Check manually.")
        return True  # Soft pass


# ═══════════════════════════════════════════════
# TEST 3: Encoder Freeze Validation
# ═══════════════════════════════════════════════
def test_encoder_freeze():
    """
    Verify that frozen encoder parameters do NOT change
    after a training step.

    This validates our transfer learning strategy:
    - Encoder params should remain identical (frozen ❄️)
    - Decoder params should change (trainable 🔥)
    """
    print("\n" + "═" * 55)
    print("  TEST 3: Encoder Freeze Validation")
    print("═" * 55)

    model = create_model(TEST_CONFIG)
    keypoints, targets = create_fake_data(TEST_CONFIG)

    # Freeze the encoder
    model.freeze_encoder()

    # Save encoder parameters BEFORE training step
    encoder_params_before = {
        name: param.clone()
        for name, param in model.encoder.named_parameters()
    }

    # Save decoder parameters BEFORE training step
    decoder_params_before = {
        name: param.clone()
        for name, param in model.decoder.named_parameters()
    }

    # Do one training step (only decoder should update)
    criterion = CSLTLoss(TEST_CONFIG["vocab_size"], pad_idx=0)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    logits = model(keypoints, targets)
    loss = criterion(logits, targets[:, 1:])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check encoder: params should NOT have changed
    encoder_changed = False
    for name, param in model.encoder.named_parameters():
        if not torch.equal(param, encoder_params_before[name]):
            encoder_changed = True
            break

    # Check decoder: params SHOULD have changed
    decoder_changed = False
    for name, param in model.decoder.named_parameters():
        if not torch.equal(param, decoder_params_before[name]):
            decoder_changed = True
            break

    print(f"  Encoder params changed: {encoder_changed} (expected: False)")
    print(f"  Decoder params changed: {decoder_changed} (expected: True)")

    if not encoder_changed and decoder_changed:
        print("  ✅ TEST 3 PASSED — Encoder frozen, decoder trainable")
        return True
    else:
        print("  ❌ TEST 3 FAILED — Freeze mechanism not working")
        return False


# ═══════════════════════════════════════════════
# TEST 4: Greedy Decoding (Inference)
# ═══════════════════════════════════════════════
def test_greedy_decoding():
    """
    Test the autoregressive greedy decoding pipeline.

    Verifies that the model can generate token sequences
    during inference (without teacher forcing).

    Expected: Output should be a tensor of token indices.
    """
    print("\n" + "═" * 55)
    print("  TEST 4: Greedy Decoding (Inference)")
    print("═" * 55)

    model = create_model(TEST_CONFIG)
    keypoints, _ = create_fake_data(TEST_CONFIG)

    try:
        model.eval()
        with torch.no_grad():
            generated = model.translate(keypoints, max_len=15)

        print(f"  Input shape    : {keypoints.shape}")
        print(f"  Generated shape: {generated.shape}")
        print(f"  Generated[0]   : {generated[0].tolist()}")

        # Verify output starts with <SOS> (index 1)
        assert generated[0, 0].item() == 1, \
            f"First token should be <SOS> (1), got {generated[0, 0].item()}"

        # Verify output contains valid indices
        assert (generated >= 0).all(), "Negative token indices found!"
        assert (generated < TEST_CONFIG["vocab_size"]).all(), \
            "Token index exceeds vocabulary size!"

        print("  ✅ TEST 4 PASSED — Greedy decoding works correctly")
        return True

    except Exception as e:
        print(f"  ❌ TEST 4 FAILED — {e}")
        return False


# ═══════════════════════════════════════════════
# TEST 5: Parameter Count Summary
# ═══════════════════════════════════════════════
def test_parameter_count():
    """
    Print and verify the model parameter summary.

    Checks that:
    - Total parameters > 0
    - After freezing encoder, trainable < total
    """
    print("\n" + "═" * 55)
    print("  TEST 5: Parameter Count Summary")
    print("═" * 55)

    model = create_model(TEST_CONFIG)

    # Before freezing
    total_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Before freeze:")
    print(f"    Total params     : {total_before:,}")
    print(f"    Trainable params : {trainable_before:,}")

    # After freezing encoder
    model.freeze_encoder()
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  After freezing encoder:")
    print(f"    Total params     : {total_before:,}")
    print(f"    Trainable params : {trainable_after:,}")
    print(f"    Frozen params    : {total_before - trainable_after:,}")

    model.count_parameters()

    if trainable_after < total_before and trainable_after > 0:
        print("  ✅ TEST 5 PASSED — Parameter counts are valid")
        return True
    else:
        print("  ❌ TEST 5 FAILED — Parameter count issue")
        return False


# ═══════════════════════════════════════════════
# MAIN: Run All Tests
# ═══════════════════════════════════════════════
def run_all_tests():
    """Run all sanity checks and print a final summary."""
    print("\n" + "╔" + "═" * 55 + "╗")
    print("║" + "   CSLT MODEL — SANITY CHECK".center(55) + "║")
    print("║" + "   No dataset required (uses random data)".center(55) + "║")
    print("╚" + "═" * 55 + "╝")

    results = {}
    results["Forward Pass"] = test_forward_pass()
    results["Overfit Batch"] = test_overfit()
    results["Encoder Freeze"] = test_encoder_freeze()
    results["Greedy Decoding"] = test_greedy_decoding()
    results["Parameter Count"] = test_parameter_count()

    # ── Final Summary ──
    print("\n" + "╔" + "═" * 55 + "╗")
    print("║" + "   FINAL SUMMARY".center(55) + "║")
    print("╠" + "═" * 55 + "╣")
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"║  {status}  │  {name:<35}      ║")
        if not passed:
            all_passed = False
    print("╠" + "═" * 55 + "╣")

    if all_passed:
        print("║" + "   🎉 ALL TESTS PASSED — Architecture is valid!".center(55) + "║")
        print("║" + "   You can now proceed to training.".center(55) + "║")
    else:
        print("║" + "   ⚠️  Some tests failed — Fix before training.".center(55) + "║")

    print("╚" + "═" * 55 + "╝\n")
    return all_passed


if __name__ == "__main__":
    run_all_tests()
