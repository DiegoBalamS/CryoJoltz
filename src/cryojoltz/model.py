from pathlib import Path
from typing import Optional

import jax.numpy as jnp
from boltz.main import (
    BoltzDiffusionParams,
    BoltzInferenceDataModule,
    BoltzProcessedInput,
    check_inputs,
    process_inputs,
    download_boltz1 as download,
)
from boltz.model.models.boltz1 import Boltz1
import joltz


class JoltzTrainer:
    def __init__(
        self,
        fasta_path: Path,
        out_dir: Path = Path("test_prediction"),
        cache_dir: Path = Path("~/.boltz").expanduser(),
        recycling_steps: int = 1,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        use_msa_server: bool = True,
        msa_server_url: str = "https://api.colabfold.com",
        msa_pairing_strategy: str = "greedy",
    ):
        self.fasta_path = fasta_path
        self.out_dir = out_dir
        self.cache_dir = cache_dir
        self.recycling_steps = recycling_steps
        self.sampling_steps = sampling_steps
        self.diffusion_samples = diffusion_samples
        self.use_msa_server = use_msa_server
        self.msa_server_url = msa_server_url
        self.msa_pairing_strategy = msa_pairing_strategy

        self.joltz_model = None
        self.jax_features = None

    def setup(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        download(self.cache_dir)

        # Load torch model
        torch_model = Boltz1.load_from_checkpoint(
            self.cache_dir / "boltz1_conf.ckpt",
            strict=True,
            map_location="cpu",
            predict_args={
                "recycling_steps": self.recycling_steps,
                "sampling_steps": self.sampling_steps,
                "diffusion_samples": self.diffusion_samples,
                "write_confidence_summary": True,
                "write_full_pae": True,
                "write_full_pde": True,
            },
            diffusion_process_args=BoltzDiffusionParams().__dict__,
            ema=False,
        )

        # Convert to JAX
        self.joltz_model = joltz.from_torch(torch_model)

        # Process input
        data = check_inputs(self.fasta_path)
        manifest = process_inputs(
            data=data,
            out_dir=self.out_dir,
            ccd_path=self.cache_dir / "ccd.pkl",
            mol_dir=self.cache_dir / "mols",
            use_msa_server=self.use_msa_server,
            msa_server_url=self.msa_server_url,
            msa_pairing_strategy=self.msa_pairing_strategy,
        )

        processed = BoltzProcessedInput(
            manifest=manifest.load(self.out_dir / "processed/manifest.json"),
            targets_dir=self.out_dir / "processed/structures",
            msa_dir=self.out_dir / "processed/msa",
        )

        data_module = BoltzInferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            num_workers=0,
        )

        features_dict = list(data_module.predict_dataloader())[0]
        self.jax_features = {
            k: jnp.array(v) for k, v in features_dict.items() if k != "record"
        }

    def get_model_and_features(self):
        if self.joltz_model is None or self.jax_features is None:
            raise RuntimeError("You must call setup() before accessing model or features.")
        return self.joltz_model, self.jax_features


