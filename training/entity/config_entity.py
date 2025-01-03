from dataclasses import dataclass
from pathlib import Path 

#1
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source: Path
    data_dir: Path
    STATUS_FILE: str
#2
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_dir: Path
    all_schema: dict
    STATUS_FILE: str

#5
@dataclass(frozen=True)
class FeatureEngineeringConfig:
    root_dir: Path
    source: Path
    data_dir: Path
    STATUS_FILE: str

@dataclass(frozen=True)
class FeatureExtractionConfig:
    root_dir: Path
    source: Path
    transform_data: Path
    target: Path
    STATUS_FILE: str


@dataclass(frozen=True)
class CrossValConfig:
    root_dir: Path
    transform_data: Path
    target: Path
    final_train_data_path: Path
    final_test_data_path: Path
    best_model_params: Path
    STATUS_FILE: str
    

#6
# Changes will be made as per the model is configured
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    final_train_data_path: Path
    final_test_data_path: Path
    best_model_params: Path
    STATUS_FILE: str
#7
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    STATUS_FILE: str