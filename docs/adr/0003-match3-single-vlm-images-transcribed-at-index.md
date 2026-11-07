# Match 3: one vision-language model, images transcribed during indexing

Match 3 Questions can be answered only from text printed inside images (2 of the 10 sample
Questions), the Harness forbids network access, and the base image already uses 29.2 of the
60 GiB image limit. We ship a single vision-language model (Qwen3-VL-8B-Instruct, bf16, ~17 GB)
behind the resident model server instead of a text LLM plus a separate OCR model: one set of
weights fits the size budget, one process holds the GPU, and the same model answers Questions.

Every png/jpg is transcribed to text once during `--index` (charged to the 10-minute startup
budget) and stored as an ordinary Corpus document cited by the image's path. Questions then run
the same text retrieval for every file type and stay inside the 30-second limit. The cost is a
longer index pass that grows with the number of images; if a graded Corpus has too many images
to transcribe in time, transcribe in rank order and fall back to on-demand transcription.
