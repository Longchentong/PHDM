import os

import torch
import transformers

try:
    import safetensors.torch
except ImportError:
    safetensors = None

from peft.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, set_peft_model_state_dict


class PeftAdapterCheckpointTrainer(transformers.Trainer):
    def _active_adapter_name(self, model):
        if hasattr(model, "active_adapters"):
            active_adapters = model.active_adapters
            if callable(active_adapters):
                active_adapters = active_adapters()
            if active_adapters:
                return active_adapters[0]

        active_adapter = getattr(model, "active_adapter", None)
        if callable(active_adapter):
            active_adapter = active_adapter()
        return active_adapter or "default"

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = self.accelerator.unwrap_model(self.model)
        if hasattr(model, "peft_config") and hasattr(model, "save_pretrained"):
            model.save_pretrained(
                output_dir,
                safe_serialization=self.args.save_safetensors,
            )
        else:
            super()._save(output_dir=output_dir, state_dict=state_dict)
            return

        processing_class = getattr(self, "processing_class", None)
        if processing_class is not None:
            processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            self.data_collator.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_best_model(self):
        best_checkpoint = getattr(self.state, "best_model_checkpoint", None)
        if best_checkpoint:
            adapter_bin = os.path.join(best_checkpoint, WEIGHTS_NAME)
            adapter_safe = os.path.join(best_checkpoint, SAFETENSORS_WEIGHTS_NAME)
            if os.path.exists(adapter_safe) or os.path.exists(adapter_bin):
                wrapped_model = getattr(self, "model_wrapped", None) or self.model
                model = self.accelerator.unwrap_model(wrapped_model)
                model = getattr(model, "_orig_mod", model)
                adapter_name = self._active_adapter_name(model)

                if hasattr(model, "load_adapter"):
                    load_result = model.load_adapter(
                        best_checkpoint,
                        adapter_name,
                        is_trainable=True,
                    )
                    if load_result is not None:
                        self._issue_warnings_after_load(load_result)
                    return

                if hasattr(model, "peft_config"):
                    if os.path.exists(adapter_safe):
                        if safetensors is None:
                            raise ImportError(
                                f"Cannot load {adapter_safe} because safetensors is not installed."
                            )
                        state_dict = safetensors.torch.load_file(adapter_safe, device="cpu")
                    else:
                        state_dict = torch.load(
                            adapter_bin,
                            map_location="cpu",
                            weights_only=True,
                        )
                    load_result = set_peft_model_state_dict(
                        model,
                        state_dict,
                        adapter_name=adapter_name,
                    )
                    self._issue_warnings_after_load(load_result)
                    return

        super()._load_best_model()
