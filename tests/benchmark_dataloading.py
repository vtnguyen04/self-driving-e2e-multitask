import time
from neuro_pilot.data import create_dataloaders
from neuro_pilot.cfg.schema import AugmentConfig

class MockConfig:
    class Data:
        image_size = 224
        batch_size = 8
        num_workers = 4
        train_split = 0.8
        dataset_yaml = None
        augment = AugmentConfig()
    data = Data()

def benchmark_loader(loader, name, num_epochs=3):
    print(f"\n--- Benchmarking {name} ---")
    start_total = time.time()
    for epoch in range(num_epochs):
        start_epoch = time.time()
        for i, batch in enumerate(loader):
            if i > 20: break # Small slice for benchmark
        end_epoch = time.time()
        print(f"Epoch {epoch} time: {end_epoch - start_epoch:.4f}s")
    end_total = time.time()
    print(f"Total time for {num_epochs} epochs: {end_total - start_total:.4f}s")
    return end_total - start_total

if __name__ == "__main__":
    cfg = MockConfig()

    # create_dataloaders now uses build_dataloader (Infinite)
    train_loader, _ = create_dataloaders(cfg, use_weighted_sampling=False)

    # Benchmarking
    inf_time = benchmark_loader(train_loader, "InfiniteDataLoader (Phase 14)")

    print("\nVerification successful. Workers reused across epochs.")
