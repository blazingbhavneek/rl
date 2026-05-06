# GRPO TODOs

## pipeline/grpo/grpo.py

### Cleanup
- [ ] Write docstring with correct flow for `GRPOPipeline` class
- [ ] Clean up vLLM engine init — many hardcodings
- [ ] Add comments for each `_backward_pass` parameter
- [ ] Move `_score` into verifier to make pipeline task-agnostic
- [ ] Make system prompt model-agnostic (hardcoded Gemma `<|think|>`)
- [ ] Handle generation logging better — not each step in a separate file

### Research / Investigation
- [ ] Investigate whether restoring RNG state actually matters for RL continuations
- [ ] Research what gradient clipping does and how it affects this training process
- [ ] Investigate gradient accumulation — would it improve resilience to degradation?
- [ ] Research whether a separate SFT optimizer is correct; what LR it should have
- [ ] Understand why `model.eval()` is needed before saving LoRA adapter
- [ ] Investigate why `next(model.parameters())` / active LoRA adapter is needed in `_backward_pass`

### Refactoring
- [ ] Make a helper function for saving LoRA adapters with metadata (used in buffer save + permanent checkpoint)
- [ ] Decouple pipeline from algo — too tightly coupled for GRPO variants
- [ ] Simplify GRPO per-problem abstraction — passing train_model + rows each batch seems redundant

### Teacher Refinement
- [ ] Streamline RL + SFT steps — unnecessary breakdown currently
- [ ] Activate `engine.sleep()` during backprop for larger models

---

## algo/grpo.py

---

## pipeline/grpo/teacher.py

- [ ] Fix `error_block` double-truncation — `[:1500] + [-1500:]` duplicates content when string is shorter than 3000 chars
