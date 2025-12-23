# ML Training & Deployment Flow - Mermaid Sequence Diagram

```mermaid
sequenceDiagram
    actor DataPartner
    participant AltaStata@{ "type": "collections" }
    participant ModelTraining
    Note right of ModelTraining: Operates inside CoCo
    participant ModelInference
    Note right of ModelInference: Operates inside CoCo
    actor DataScientist
    Note over DataPartner,AltaStata: Data Ingestion
    DataPartner->>AltaStata: Encrypt & Upload Training Dataset
    Note over DataScientist,ModelTraining: Model Training and Store
    DataScientist->>ModelTraining: Initiate model training
    ModelTraining->>AltaStata: Access training dataset
    ModelTraining->>ModelTraining: Train ML Model
    ModelTraining->>AltaStata: Encrypt & Store ML Model
    Note over DataScientist,ModelInference: Model Deployment & Inference
    DataScientist->>ModelInference: Send Query
    ModelInference->>AltaStata: Read ML Model
    ModelInference->>ModelInference: Process Query
    ModelInference->>DataScientist: Return Predictions
```

## Participants:
1. **Data Partner** - Uploads training datasets
2. **Data Scientist** - Sends queries to deployed model
3. **AltaStata** - Encrypted storage for datasets and models
4. **Model Training** - PyTorch/TensorFlow for model training
5. **Model Inference** - Handles model deployment and inference processing

## Key Features:
- ✅ Three distinct phases: Data Ingestion → Model Training → Model Deployment & Inference
- ✅ Encrypted storage at every stage (datasets and models)
- ✅ Direct access: ModelTraining streams from AltaStata during training
- ✅ Model lifecycle: train → store → deploy → infer
- ✅ DataScientist queries deployed model through ModelInference

