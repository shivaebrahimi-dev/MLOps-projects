# MLOps Course Glossary

Short definitions of terms used across the course projects, written by the student.

**Experiment tracking**: Recording the parameters, metrics and artifacts of every training run so that runs can be
compared and reproduced. The Fashion-MNIST project uses MLflow for this.

**Object storage**: Storing files as objects in buckets behind an S3-compatible API. Both the Fashion-MNIST project
and the zero-shot platform use MinIO: the first stores trained model files, the second stores classification
results as JSON records.

**Webhook**: An HTTP request that one system sends to another when an event happens. The Fashion-MNIST trainer sends
a training-complete webhook only after the model has been uploaded.

**Zero-shot classification**: Assigning text to labels that the model was not trained on, by framing each label as a
natural-language inference hypothesis. The zero-shot platform uses DeBERTa and BART MNLI models for this.

**Specification-driven development**: Writing the requirements, design and task list before the code. Spec Kit
produces a constitution, a specification, a plan and a task list, and the code is checked against them.

**Retrieval-Augmented Generation (RAG)**: Answering a question by first retrieving relevant passages from a
knowledge base and then asking a language model to answer from those passages only, with citations.

**Hallucination**: A model output that states something not supported by its input or evidence. In RAG it is
measured as the share of claims that the retrieved context does not support.

**Observability**: Recording what a system did, step by step, so that failures and slow steps can be found. For LLM
applications this usually means traces with nested spans for each model call and tool call.

**Data drift**: A change in the distribution of production inputs compared with the training data. It is often
detected with the Population Stability Index (PSI) or the Kolmogorov–Smirnov (KS) test.
