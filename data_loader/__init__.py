from torch.utils.data import DataLoader


def get_dataloader(cfg: dict):
    """Return train, val, test DataLoaders and the train dataset (possibly InfoBatch-wrapped)."""
    method = cfg["data"]["split"]["method"]
    if method == "random":
        from .random_split import RandomSplitDataset as DS
    elif method == "by_state":
        from .by_state_split import StateSplitDataset as DS
    elif method == "by_climate":
        from .by_climate_split import ClimateSplitDataset as DS
    elif method == "by_tree_type":
        from .by_tree_split import TreeSplitDataset as DS
    else:
        raise ValueError(f"Unknown split method: {method}")

    train_ds = DS(cfg, split="train")
    val_ds   = DS(cfg, split="val")
    test_ds  = DS(cfg, split="test")

    # Optionally wrap train dataset with InfoBatch
    ib_cfg = cfg.get('infobatch', {})
    use_infobatch = ib_cfg.get('enabled', False)
    train_sampler = None
    train_shuffle = True

    if use_infobatch:
        print("Using InfoBatch for training")
        from infobatch import InfoBatch
        num_epochs = cfg['training']['epochs']
        prune_ratio = ib_cfg.get('prune_ratio', 0.5)
        delta = ib_cfg.get('delta', 0.875)
        train_ds = InfoBatch(train_ds, num_epochs=num_epochs, prune_ratio=prune_ratio, delta=delta)
        train_sampler = train_ds.sampler
        train_shuffle = False  # sampler handles ordering

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training'].get("batch_size", 32),
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=cfg['experiment'].get("num_workers", 16),
        pin_memory=True,
        prefetch_factor=2)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training'].get("batch_size", 32),
        shuffle=False,
        num_workers=cfg['experiment'].get("num_workers", 16),
        pin_memory=True,
        prefetch_factor=2)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg['training'].get("batch_size", 32),
        shuffle=False,
        num_workers=cfg['experiment'].get("num_workers", 16),
        pin_memory=True,
        prefetch_factor=2)

    return train_loader, val_loader, test_loader, train_ds
